
from object_detection.inputs import get_reduce_to_frame_fn, augment_input_data, transform_input_data, pad_input_data_to_static_shapes, _get_features_dict, _get_labels_dict
from object_detection.builders.dataset_builder import read_dataset, shard_function_for_context
from object_detection.builders import dataset_builder, decoder_builder, model_builder, preprocessor_builder, image_resizer_builder
from object_detection.utils import config_util
from object_detection.protos import eval_pb2, image_resizer_pb2, input_reader_pb2, model_pb2, train_pb2
import functools
import sys
import tensorflow as tf


@tf.function
def intermediate_parse(dataset, imagesets):

    def predicate(element):
        feature_description = {
            'image/imageset': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            'image/object/class/text': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/object/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}
        parsed_example = tf.io.parse_example(element, feature_description)
        imageset = parsed_example['image/imageset']
        return in_list(imageset)

    def in_list(imgset):
      b = False
      for imageset in imagesets:
        if imgset == imageset:
          b = True
      return b

    dataset = dataset.filter(predicate)
    return dataset
    

def build(imagesets, input_reader_config, batch_size=None, transform_input_data_fn=None,
          input_context=None, reduce_to_frame_fn=None):
  """Builds a tf.data.Dataset.
  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.
  Args:
    imagesets: A list of imagesets from which to create the dataset
    input_reader_config: A input_reader_pb2.InputReader object.
    batch_size: Batch size. If batch size is None, no batching is performed.
    transform_input_data_fn: Function to apply transformation to all records,
      or None if no extra decoding is required.
    input_context: optional, A tf.distribute.InputContext object used to
      shard filenames and compute per-replica batch_size when this function
      is being called per-replica.
    reduce_to_frame_fn: Function that extracts frames from tf.SequenceExample
      type input data.
  Returns:
    A tf.data.Dataset based on the input_reader_config.
  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  decoder = decoder_builder.build(input_reader_config)

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')
    def dataset_map_fn(dataset, fn_to_map, batch_size=None,
                       input_reader_config=None):
      """Handles whether or not to use the legacy map function.
      Args:
        dataset: A tf.Dataset.
        fn_to_map: The function to be mapped for that dataset.
        batch_size: Batch size. If batch size is None, no batching is performed.
        input_reader_config: A input_reader_pb2.InputReader object.
      Returns:
        A tf.data.Dataset mapped with fn_to_map.
      """
      if hasattr(dataset, 'map_with_legacy_function'):
        if batch_size:
          num_parallel_calls = batch_size * (
              input_reader_config.num_parallel_batches)
        else:
          num_parallel_calls = input_reader_config.num_parallel_map_calls
        dataset = dataset.map_with_legacy_function(
            fn_to_map, num_parallel_calls=num_parallel_calls)
      else:
        dataset = dataset.map(fn_to_map, tf.data.experimental.AUTOTUNE)
      return dataset
    shard_fn = shard_function_for_context(input_context)
    if input_context is not None:
      batch_size = input_context.get_per_replica_batch_size(batch_size)
    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        config.input_path[:], input_reader_config, filename_shard_fn=shard_fn)
    if input_reader_config.sample_1_of_n_examples > 1:
      dataset = dataset.shard(input_reader_config.sample_1_of_n_examples, 0)
 
    dataset = intermediate_parse(dataset, imagesets)
    dataset = dataset_map_fn(dataset, decoder.decode, batch_size,
                             input_reader_config)

    if reduce_to_frame_fn:
      dataset = reduce_to_frame_fn(dataset, dataset_map_fn, batch_size,
                                   input_reader_config)
    if transform_input_data_fn is not None:
      dataset = dataset_map_fn(dataset, transform_input_data_fn,
                               batch_size, input_reader_config)
    if batch_size:
      #NOTE: set drop_remainder to True because of attribute error
      dataset = dataset.batch(batch_size,
                              drop_remainder=True)
    dataset = dataset.prefetch(input_reader_config.num_prefetch_batches)
    return dataset

  raise ValueError('Unsupported input_reader_config.')

def train_input(imagesets, train_config, train_input_config,
                model_config, model=None, params=None, input_context=None):
  """Returns `features` and `labels` tensor dictionaries for training.
  Args:
    imagesets: A list of imagesets from which to create the dataset
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.
    model: A pre-constructed Detection Model.
      If None, one will be created from the config.
    params: Parameter dictionary passed from the estimator.
    input_context: optional, A tf.distribute.InputContext object used to
      shard filenames and compute per-replica batch_size when this function
      is being called per-replica.
  Returns:
    A tf.data.Dataset that holds (features, labels) tuple.
    features: Dictionary of feature tensors.
      features[fields.InputDataFields.image] is a [batch_size, H, W, C]
        float32 tensor with preprocessed images.
      features[HASH_KEY] is a [batch_size] int32 tensor representing unique
        identifiers for the images.
      features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
        int32 tensor representing the true image shapes, as preprocessed
        images could be padded.
      features[fields.InputDataFields.original_image] (optional) is a
        [batch_size, H, W, C] float32 tensor with original images.
    labels: Dictionary of groundtruth tensors.
      labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
        int32 tensor indicating the number of groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_boxes] is a
        [batch_size, num_boxes, 4] float32 tensor containing the corners of
        the groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_classes] is a
        [batch_size, num_boxes, num_classes] float32 one-hot tensor of
        classes.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes] float32 tensor containing groundtruth weights
        for the boxes.
      -- Optional --
      labels[fields.InputDataFields.groundtruth_instance_masks] is a
        [batch_size, num_boxes, H, W] float32 tensor containing only binary
        values, which represent instance masks for objects.
      labels[fields.InputDataFields.groundtruth_keypoints] is a
        [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
        keypoints for each box.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes, num_keypoints] float32 tensor containing
        groundtruth weights for the keypoints.
      labels[fields.InputDataFields.groundtruth_visibilities] is a
        [batch_size, num_boxes, num_keypoints] bool tensor containing
        groundtruth visibilities for each keypoint.
      labels[fields.InputDataFields.groundtruth_labeled_classes] is a
        [batch_size, num_classes] float32 k-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_dp_num_points] is a
        [batch_size, num_boxes] int32 tensor with the number of sampled
        DensePose points per object.
      labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
        [batch_size, num_boxes, max_sampled_points] int32 tensor with the
        DensePose part ids (0-indexed) per object.
      labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
        [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
        DensePose surface coordinates. The format is (y, x, v, u), where (y, x)
        are normalized image coordinates and (v, u) are normalized surface part
        coordinates.
      labels[fields.InputDataFields.groundtruth_track_ids] is a
        [batch_size, num_boxes] int32 tensor with the track ID for each object.
  Raises:
    TypeError: if the `train_config`, `train_input_config` or `model_config`
      are not of the correct type.
  """
  if not isinstance(train_config, train_pb2.TrainConfig):
    raise TypeError('For training mode, the `train_config` must be a '
                    'train_pb2.TrainConfig.')
  if not isinstance(train_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `train_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if model is None:
    model_preprocess_fn = model_builder.build(
        model_config, is_training=True).preprocess
  else:
    model_preprocess_fn = model.preprocess

  num_classes = config_util.get_number_of_classes(model_config)

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]
    data_augmentation_fn = functools.partial(
        augment_input_data,
        data_augmentation_options=data_augmentation_options)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    keypoint_type_weight = train_input_config.keypoint_type_weight or None
    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=data_augmentation_fn,
        merge_multiple_boxes=train_config.merge_multiple_label_boxes,
        retain_original_image=train_config.retain_original_images,
        use_multiclass_scores=train_config.use_multiclass_scores,
        use_bfloat16=train_config.use_bfloat16,
        keypoint_type_weight=keypoint_type_weight)

    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=train_input_config.max_number_of_boxes,
        num_classes=num_classes,
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config),
        max_num_context_features=config_util.get_max_num_context_features(
            model_config),
        context_feature_length=config_util.get_context_feature_length(
            model_config))
    include_source_id = train_input_config.include_source_id
    return (_get_features_dict(tensor_dict, include_source_id),
            _get_labels_dict(tensor_dict))
  reduce_to_frame_fn = get_reduce_to_frame_fn(train_input_config, True)

  dataset = build(
      imagesets, 
      train_input_config,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      batch_size=params['batch_size'] if params else train_config.batch_size,
      input_context=input_context,
      reduce_to_frame_fn=reduce_to_frame_fn)
  return dataset

def eval_input(imagesets, eval_config, eval_input_config, model_config,
               model=None, params=None, input_context=None):
  """Returns `features` and `labels` tensor dictionaries for evaluation.
  Args:
    imagesets: A list of imagesets from which to create the dataset
    eval_config: An eval_pb2.EvalConfig.
    eval_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.
    model: A pre-constructed Detection Model.
      If None, one will be created from the config.
    params: Parameter dictionary passed from the estimator.
    input_context: optional, A tf.distribute.InputContext object used to
      shard filenames and compute per-replica batch_size when this function
      is being called per-replica.
  Returns:
    A tf.data.Dataset that holds (features, labels) tuple.
    features: Dictionary of feature tensors.
      features[fields.InputDataFields.image] is a [1, H, W, C] float32 tensor
        with preprocessed images.
      features[HASH_KEY] is a [1] int32 tensor representing unique
        identifiers for the images.
      features[fields.InputDataFields.true_image_shape] is a [1, 3]
        int32 tensor representing the true image shapes, as preprocessed
        images could be padded.
      features[fields.InputDataFields.original_image] is a [1, H', W', C]
        float32 tensor with the original image.
    labels: Dictionary of groundtruth tensors.
      labels[fields.InputDataFields.groundtruth_boxes] is a [1, num_boxes, 4]
        float32 tensor containing the corners of the groundtruth boxes.
      labels[fields.InputDataFields.groundtruth_classes] is a
        [num_boxes, num_classes] float32 one-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_area] is a [1, num_boxes]
        float32 tensor containing object areas.
      labels[fields.InputDataFields.groundtruth_is_crowd] is a [1, num_boxes]
        bool tensor indicating if the boxes enclose a crowd.
      labels[fields.InputDataFields.groundtruth_difficult] is a [1, num_boxes]
        int32 tensor indicating if the boxes represent difficult instances.
      -- Optional --
      labels[fields.InputDataFields.groundtruth_instance_masks] is a
        [1, num_boxes, H, W] float32 tensor containing only binary values,
        which represent instance masks for objects.
      labels[fields.InputDataFields.groundtruth_weights] is a
        [batch_size, num_boxes, num_keypoints] float32 tensor containing
        groundtruth weights for the keypoints.
      labels[fields.InputDataFields.groundtruth_visibilities] is a
        [batch_size, num_boxes, num_keypoints] bool tensor containing
        groundtruth visibilities for each keypoint.
      labels[fields.InputDataFields.groundtruth_group_of] is a [1, num_boxes]
        bool tensor indicating if the box covers more than 5 instances of the
        same class which heavily occlude each other.
      labels[fields.InputDataFields.groundtruth_labeled_classes] is a
        [num_boxes, num_classes] float32 k-hot tensor of classes.
      labels[fields.InputDataFields.groundtruth_dp_num_points] is a
        [batch_size, num_boxes] int32 tensor with the number of sampled
        DensePose points per object.
      labels[fields.InputDataFields.groundtruth_dp_part_ids] is a
        [batch_size, num_boxes, max_sampled_points] int32 tensor with the
        DensePose part ids (0-indexed) per object.
      labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
        [batch_size, num_boxes, max_sampled_points, 4] float32 tensor with the
        DensePose surface coordinates. The format is (y, x, v, u), where (y, x)
        are normalized image coordinates and (v, u) are normalized surface part
        coordinates.
      labels[fields.InputDataFields.groundtruth_track_ids] is a
        [batch_size, num_boxes] int32 tensor with the track ID for each object.
  Raises:
    TypeError: if the `eval_config`, `eval_input_config` or `model_config`
      are not of the correct type.
  """
  params = params or {}
  if not isinstance(eval_config, eval_pb2.EvalConfig):
    raise TypeError('For eval mode, the `eval_config` must be a '
                    'train_pb2.EvalConfig.')
  if not isinstance(eval_input_config, input_reader_pb2.InputReader):
    raise TypeError('The `eval_input_config` must be a '
                    'input_reader_pb2.InputReader.')
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise TypeError('The `model_config` must be a '
                    'model_pb2.DetectionModel.')

  if eval_config.force_no_resize:
    arch = model_config.WhichOneof('model')
    arch_config = getattr(model_config, arch)
    image_resizer_proto = image_resizer_pb2.ImageResizer()
    image_resizer_proto.identity_resizer.CopyFrom(
        image_resizer_pb2.IdentityResizer())
    arch_config.image_resizer.CopyFrom(image_resizer_proto)

  if model is None:
    model_preprocess_fn = INPUT_BUILDER_UTIL_MAP['model_build'](
        model_config, is_training=False).preprocess
  else:
    model_preprocess_fn = model.preprocess

  def transform_and_pad_input_data_fn(tensor_dict):
    """Combines transform and pad operation."""
    num_classes = config_util.get_number_of_classes(model_config)

    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)
    keypoint_type_weight = eval_input_config.keypoint_type_weight or None

    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model_preprocess_fn,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None,
        retain_original_image=eval_config.retain_original_images,
        retain_original_image_additional_channels=
        eval_config.retain_original_image_additional_channels,
        keypoint_type_weight=keypoint_type_weight)
    tensor_dict = pad_input_data_to_static_shapes(
        tensor_dict=transform_data_fn(tensor_dict),
        max_num_boxes=eval_input_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config),
        max_num_context_features=config_util.get_max_num_context_features(
            model_config),
        context_feature_length=config_util.get_context_feature_length(
            model_config))
    include_source_id = eval_input_config.include_source_id
    return (_get_features_dict(tensor_dict, include_source_id),
            _get_labels_dict(tensor_dict))

  reduce_to_frame_fn = get_reduce_to_frame_fn(eval_input_config, False)

  dataset = build(
      imagesets, 
      eval_input_config,
      batch_size=params['batch_size'] if params else eval_config.batch_size,
      transform_input_data_fn=transform_and_pad_input_data_fn,
      input_context=input_context,
      reduce_to_frame_fn=reduce_to_frame_fn)
  return dataset

def testparse(proto):

    feature_description = {
        'image/imageset': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/object/keypoints': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/pose': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'image/object/translation': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)}
    parsed_example = tf.io.parse_single_example(proto, feature_description)
    imageset = parsed_example['image/imageset']

    return imageset