import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path
import itertools

import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from ravenml_tf_instance.validation.classes import DetectedClass, TruthClass


def get_num_classes(label_path):
    with open(label_path, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)

    return num_classes


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_categories(label_path: str):
    label_map = label_map_util.load_labelmap(label_path)
    num_classes = get_num_classes(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)


    return category_index


def get_image_paths(dev_path):
    image_paths = []
    mask_paths = []
    metadata_paths = []

    for item in os.listdir(dev_path):
        if item.startswith('image'):
            image_paths.append(os.path.join(dev_path, item))
            
        if len(image_paths) == 20:
            break

    for impath in image_paths:

        imname = impath.split('/')[-1]
        uid = imname.split('_')[1].split('.')[0]
        mask_name = "mask_{}.png".format(uid)
        meta_name = "meta_{}.json".format(uid)
        
        mask_paths.append(os.path.join(dev_path, mask_name))
        metadata_paths.append(os.path.join(dev_path, meta_name))

    return image_paths, mask_paths, metadata_paths


def load_images_from_paths(image_paths):
    images = []
    for impath in image_paths:
        image = Image.open(impath)
        image_np = load_image_into_numpy_array(image)

        images.append(image_np)

    return images


def load_masks_from_paths(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        mask_np = load_image_into_numpy_array(mask)

        masks.append(mask_np)

    return masks


def get_defualt_graph(model_path):

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    return detection_graph


def run_inference_for_multiple_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []
            dict_time = []
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            ostart = time.time()
            if tensor_dict.get('detection_masks') is not None:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, images[0].shape[0], images[0].shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            count = 1
            for image in images:
                # Run inference
                start = time.time()
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
                end = time.time()
                #print('inference time : {}'.format(end - start))
 
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
 
                output_dict_array.append(output_dict)
                dict_time.append(end - start)

                if count % 25 == 0:
                    print("{} images done".format(str(count)))

                count += 1


    return output_dict_array, dict_time


def convert_inference_output_to_detected_objects(category_index, outputs):
    all_detections = []
    for output in outputs:

        detections = {}
        for i in range(len(output['detection_scores'])):
            score = output['detection_scores'][i]
            class_id = output['detection_classes'][i]
            if score > 0.5 and detections.get(class_id) is None:
                class_name = category_index[class_id]['name']
                box = output['detection_boxes'][i]
                mask = output['detection_masks'][i]
                detections[class_id] = DetectedClass(class_id, class_name, round(100 * score, 3), box, mask)


        all_detections.append(detections)


    return all_detections


def visualize_and_save(images, all_detections, output_path):

    for image_idx in range(len(all_detections)):


        dirpath = Path(output_path) / 'viz' / str(image_idx)
        os.makedirs(dirpath, exist_ok=True)

        for class_id in all_detections[image_idx]:
            data = all_detections[image_idx][class_id]
            fig = plt.figure()
            
            if data.centroid is not None:
                plt.axis('off')
                plt.imshow(data.mask)
                plt.imshow(images[image_idx], cmap='jet', alpha=0.6)
                plt.scatter(data.centroid[1], data.centroid[0], color='r')
                plt.text(7, 15, data.class_name + ': ' + str(round(100 * data.score, 2)), c='w')
                
                figname = "_".join(i for i in data.class_name.lower().split(" ")) + '.png'
                figpath = dirpath / figname

                plt.savefig(figpath, bbox_inches='tight', pad_inches=0)


def get_truth_masks(masks, category_index):
    
    colors = {1:[192, 196, 207], 2:[194, 196, 1], 3:[198, 1, 10], 4:[1, 200, 25]}
    x = [-3, -2 -1, 0, 1, 2, 3]
    iters = [p for p in itertools.product(x, repeat=3)]

    all_truths = []

    for mask in masks:
        truths = {}
        for class_id in category_index:
            color = colors[class_id]
            matched = False
            empty_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
            for i in iters:
                c = np.add(color, np.array(i))
                match = np.where((mask == c).all(axis=2))
                y, x = match
                if len(y) != 0 and len(x) != 0:

                    empty_mask[match] = 1
                    matched = True

            if matched:
                truth = TruthClass(class_id, category_index[class_id]['name'], empty_mask)
                truths[class_id] = truth

        all_truths.append(truths)
    
    return all_truths
