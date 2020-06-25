import tensorflow as tf
import numpy as np
import os
from . import utils


class PoseErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, ref_points, cropsize, focal_length, experiment=None):
        super().__init__()
        self.ref_points = ref_points
        self.crop_size = cropsize
        self.focal_length = focal_length
        self.experiment = experiment
        self.targets = tf.Variable(0.0, shape=tf.TensorShape(None))
        self.outputs = tf.Variable(0.0, shape=tf.TensorShape(None))
        self.train_errors = []
        self.test_errors = []
        self.comet_step = 0

    def assign_metric_ignore(self, y_true, y_pred):
        self.targets.assign(y_true)
        self.outputs.assign(y_pred)
        return 0

    def calc_pose_error(self):
        label = KeypointsModel.decode_label(self.targets.numpy())
        kps_batch = self.outputs.numpy()
        kps_batch = kps_batch.reshape(kps_batch.shape[0], -1, 2)
        kps_batch = kps_batch * (self.crop_size // 2) + (self.crop_size // 2)
        error_batch = np.zeros(kps_batch.shape[0])
        for i, kps in enumerate(kps_batch):
            r_vec, t_vec, _, _ = utils.calculate_pose_vectors(
                self.ref_points, kps,
                [self.focal_length, self.focal_length], [self.crop_size, self.crop_size],
                extra_crop_params={
                    'centroid': label['centroid'][i],
                    'bbox_size': label['bbox_size'][i],
                    'imdims': [label['height'][i], label['width'][i]],
                }
            )

            error_batch[i] = utils.geodesic_error(r_vec, label['pose'][i].numpy())
        return error_batch

    def on_epoch_begin(self, epoch, logs=None):
        self.train_errors = []

    def on_train_batch_end(self, batch, logs=None):
        self.comet_step += 1
        mean = np.mean(self.calc_pose_error())
        self.train_errors.append(mean)
        running_mean = np.mean(self.train_errors)
        print(f' - pose error: {running_mean:.4f} ({np.degrees(running_mean):.2f} deg)')
        if self.experiment:
            self.experiment.log_metric('pose_error_deg', np.degrees(mean), step=self.comet_step)

    def on_test_begin(self, logs=None):
        self.test_errors = []

    def on_test_batch_end(self, batch, logs=None):
        self.comet_step += 1
        self.test_errors.append(np.mean(self.calc_pose_error()))

    def on_test_end(self, logs=None):
        mean = np.mean(self.test_errors)
        print(f'Eval pose error: {mean:.4f} ({np.degrees(mean):.2f} deg)')
        if self.experiment:
            with self.experiment.validate():
                self.experiment.log_metric('pose_error_deg', np.degrees(mean), step=self.comet_step)


class KeypointsModel:

    OPTIMIZERS = {
        'SGD': tf.keras.optimizers.SGD,
        'Adam': tf.keras.optimizers.Adam,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'RMSProp': tf.keras.optimizers.RMSprop
    }

    def __init__(self, data_dir, hp, keypoints_3d):
        """
        :param data_dir: path to data directory
        :param hp: dictionary of hyperparameters
        """
        self.data_dir = data_dir
        self.hp = hp
        self.nb_keypoints = hp['keypoints']
        self.keypoints_3d = keypoints_3d[:self.nb_keypoints]
        self.crop_size = hp['crop_size']
        self.crop_size_rounded = int(2**round(np.log2(hp['crop_size'])))

    def _get_dataset(self, split_name, train):
        """
        Each item in the dataset is a tuple (image, truth).
        image is a Tensor of shape (crop_size, crop_size, 3).
        truth is a unit quaternion Tensor of shape (4,)

        :param split_name: name of split (e.g. 'train' for 'train-0000-of-0001.tfrecord')
        :param train: boolean, whether or not to perform training augmentations
        :return: a tuple (TFRecordDataset, num_examples) for that split
        """
        features = {
            'image/height':
                tf.io.FixedLenFeature([], tf.int64),
            'image/width':
                tf.io.FixedLenFeature([], tf.int64),
            'image/object/keypoints':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.io.VarLenFeature(tf.float32),
            'image/encoded':
                tf.io.FixedLenFeature([], tf.string),
            'image/object/pose':
                tf.io.FixedLenFeature([4], tf.float32)
        }

        if self.hp['model_arch'] in ['unet']:
            cropsize = self.crop_size_rounded
        else:
            cropsize = self.crop_size

        def _parse_function(example):
            parsed = tf.io.parse_single_example(example, features)

            # find an approximate bounding box to crop the image
            height = tf.cast(parsed['image/height'], tf.float32)
            width = tf.cast(parsed['image/width'], tf.float32)
            xmin = parsed['image/object/bbox/xmin'].values[0] * width
            xmax = parsed['image/object/bbox/xmax'].values[0] * width
            ymin = parsed['image/object/bbox/ymin'].values[0] * height
            ymax = parsed['image/object/bbox/ymax'].values[0] * height
            centroid = tf.stack([(ymax + ymin) / 2, (xmax + xmin) / 2], axis=-1)
            bbox_size = tf.maximum(xmax - xmin, ymax - ymin) * 1.25

            # random positioning
            if train:
                bbox_size *= tf.random.uniform([], minval=1.0, maxval=1.5)
                # size / 10 ensures that object does not go off screen
                centroid += tf.random.uniform([2], minval=-bbox_size / 10, maxval=bbox_size / 10)

            # decode, preprocess to [-1, 1] range, and crop image/keypoints
            old_dims, image = self.preprocess_image(parsed['image/encoded'], 
                centroid, bbox_size, cropsize)
            keypoints = self.preprocess_keypoints(parsed['image/object/keypoints'].values, 
                centroid, bbox_size, cropsize, old_dims, self.nb_keypoints)

            # other augmentations
            if train:
                # image values into the [0, 1] format
                image = (image + 1) / 2

                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
                image = tf.clip_by_value(image, 0, 1)

                # convert back to [-1, 1] format
                image = image * 2 - 1

            truth = self.encode_label(
                keypoints=keypoints,
                pose=parsed['image/object/pose'],
                height=height,
                width=width,
                bbox_size=bbox_size,
                centroid=centroid
            )
            return image, truth

        with open(os.path.join(self.data_dir, f"{split_name}.record.numexamples"), "r") as f:
            num_examples = int(f.read())
        filenames = tf.io.gfile.glob(os.path.join(self.data_dir, f"{split_name}.record-*"))

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
        if self.hp['cache_train_data']:
            dataset = dataset.cache()
        return dataset.map(_parse_function, num_parallel_calls=16), num_examples

    def train(self, logdir, experiment=None):
        train_dataset, num_train = self._get_dataset('train', True)
        train_dataset = train_dataset.shuffle(self.hp['shuffle_buffer_size']).batch(self.hp['batch_size']).repeat()
        if self.hp['prefetch_num_batches']:
            train_dataset = train_dataset.prefetch(self.hp['prefetch_num_batches'])
        val_dataset, num_val = self._get_dataset('test', False)
        val_dataset = val_dataset.batch(self.hp['batch_size']).repeat()

        # imgs, kps = list(train_dataset.take(1).as_numpy_iterator())[0]
        # img, kp = img[0], kps[0]
        # img = ((img + 1) / 2 * 255).astype(np.uint8)
        # w, h, _ = img.shape
        # for i in range(len(kp) // 2):
        #     y = int(kp[i * 2] * (w // 2) + (w // 2))
        #     x = int(kp[i * 2 + 1] * (w // 2) + (w // 2))
        #     cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
        # cv2.imwrite('test.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        model_path = os.path.join(logdir, "model.h5")
        pose_error_callback = PoseErrorCallback(self.keypoints_3d, self.crop_size, self.hp['pnp_focal_length'], experiment)
        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=logdir,
                write_graph=False,
                profile_batch=0
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            pose_error_callback
        ]

        model = {
            'mobilenet': self._gen_model_mobilenet,
            'densenet': self._gen_model_densenet,
            'unet': self._get_custom_unet
        }[self.hp['model_arch']]()
        print(model.summary())

        optimizer = self.OPTIMIZERS[self.hp['optimizer']](**self.hp['optimizer_args'])
        model.compile(
            optimizer=optimizer,
            loss=self.mse_loss,
            metrics=[pose_error_callback.assign_metric_ignore],
            run_eagerly=False
        )
        model.fit(
            train_dataset,
            epochs=self.hp['epochs'],
            steps_per_epoch=num_train // self.hp['batch_size'],
            validation_data=val_dataset,
            validation_steps=num_val // self.hp['batch_size'],
            callbacks=callbacks
        )
        if experiment:
            experiment.log_model('model', model_path)

        return model_path

    def _gen_model_mobilenet(self):
        init_weights = self.hp.get('model_init_weights', '')
        assert init_weights in ['imagenet', '']
        keras_app_args = dict(
            include_top=False, weights=init_weights if init_weights != '' else None, 
            input_shape=(self.crop_size, self.crop_size, 3), 
            pooling='max'
        )
        keras_app_model = tf.keras.applications.MobileNetV2(**keras_app_args)
        app_in = keras_app_model.input
        app_out = keras_app_model.output
        x = app_out
        for i in range(self.hp['fc_count']):
            x = tf.keras.layers.Dense(self.hp['fc_width'], activation='relu')(x)
        x = tf.keras.layers.Dense(self.nb_keypoints * 2)(x)
        return tf.keras.models.Model(app_in, x)

    def _gen_model_densenet(self):
        init_weights = self.hp.get('model_init_weights', '')
        assert init_weights in ['imagenet', '']
        keras_app_args = dict(
            include_top=False, weights=init_weights if init_weights != '' else None, 
            input_shape=(self.crop_size, self.crop_size, 3), 
            pooling='max'
        )
        keras_app_model = tf.keras.applications.densenet.DenseNet121(**keras_app_args)
        app_in = keras_app_model.input
        app_out = keras_app_model.output
        x = app_out
        for i in range(self.hp['fc_count']):
            x = tf.keras.layers.Dense(self.hp['fc_width'], activation='relu')(x)
        x = tf.keras.layers.Dense(self.nb_keypoints * 2)(x)
        return tf.keras.models.Model(app_in, x)

    def _get_custom_unet(self):
        from keras_unet.models import custom_unet
        assert not self.hp.get('model_init_weights')
        unet_decoder_method = self.hp.get('model_unet_decoder', 'maxpool')
        unet_decoder_params = self.hp.get('model_unet_decoder_params')
        unet_params = dict(
            input_shape=(self.crop_size_rounded, self.crop_size_rounded, 3),
            use_batch_norm=False,
            num_classes=1,
            num_layers=4,
            activation='relu',
            filters=16,
            use_attention=False,
            upsample_mode='deconv',
            dropout=0.3,
            output_activation='linear'
        )
        unet_params.update(self.hp.get('model_unet_params', {}))
        unet_model = custom_unet(**unet_params)

        # Convert UNet 3d output to keypoints
        unet_in = unet_model.input
        unet_out = unet_model.output
        x = unet_out
        if unet_decoder_method == 'maxpool':
            if unet_decoder_params is None:
                pool_size = (4, 4)
            else:
                pool_size = unet_decoder_params
            x = tf.keras.layers.MaxPooling2D(pool_size)(x)
            x = tf.keras.layers.Flatten()(x)
            for i in range(self.hp['fc_count']):
                x = tf.keras.layers.Dense(self.hp['fc_width'], activation='relu')(x)
            x = tf.keras.layers.Dense(self.nb_keypoints * 2)(x)
        else:
            raise ValueError('Unsupported UNet decoder method.')
        return tf.keras.models.Model(unet_in, x)

    @staticmethod
    def mse_loss(y_true, y_pred):
        kps = KeypointsModel.decode_label(y_true)['keypoints']
        return tf.keras.losses.mse(tf.reshape(kps, [tf.shape(kps)[0], -1]), y_pred)

    @staticmethod
    def get_pose_error_metric(ref_points, focal_length=1422):
        def pose_error_metric(y_true, y_pred):
            label = KeypointsModel.decode_label(y_true)
            kps_batch = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, 2])
            kps_batch = kps_batch * (label['bbox_size'] // 2)[:, None, None] + label['centroid'][:, None, :]
            # kps_batch = kps_batch.numpy()
            error_batch = tf.zeros([tf.shape(kps_batch)[0]])
            for i, kps in enumerate(kps_batch):
                r_vec, t_vec, _, _ = utils.calculate_pose_vectors(
                    ref_points, kps,
                    focal_length, label['height'][i])
                error_batch[i] = utils.geodesic_error(r_vec, label['pose'][i])
            return error_batch

        return pose_error_metric

    @staticmethod
    def encode_label(*, keypoints, pose, height, width, bbox_size, centroid):
        if len(keypoints.shape) == 3:
            keypoints = tf.reshape(keypoints, [tf.shape(keypoints)[0], -1])
            return tf.concat((
                tf.reshape(tf.cast(pose, tf.float32), [-1, 4]),
                tf.reshape(tf.cast(height, tf.float32), [-1, 1]),
                tf.reshape(tf.cast(width, tf.float32), [-1, 1]),
                tf.reshape(tf.cast(bbox_size, tf.float32), [-1, 1]),
                tf.reshape(tf.cast(centroid, tf.float32), [-1, 2]),
                tf.cast(keypoints, tf.float32),
            ), axis=-1)
        else:
            keypoints = tf.reshape(keypoints, [-1])
            return tf.concat((
                tf.reshape(tf.cast(pose, tf.float32), [4]),
                tf.reshape(tf.cast(height, tf.float32), [1]),
                tf.reshape(tf.cast(width, tf.float32), [1]),
                tf.reshape(tf.cast(bbox_size, tf.float32), [1]),
                tf.reshape(tf.cast(centroid, tf.float32), [2]),
                tf.cast(keypoints, tf.float32),
            ), axis=-1)

    @staticmethod
    def decode_label(label):
        if len(label.shape) == 1:
            label = label[None, ...]
        return {
            'pose': tf.squeeze(label[:, :4]),
            'height': tf.squeeze(label[:, 4]),
            'width': tf.squeeze(label[:, 5]),
            'bbox_size': tf.squeeze(label[:, 6]),
            'centroid': tf.squeeze(label[:, 7:9]),
            'keypoints': tf.squeeze(tf.reshape(label[:, 9:], [tf.shape(label)[0], -1, 2])),
        }

    @staticmethod
    def preprocess_keypoints(parsed_kps, centroid, bbox_size, cropsize, img_size, nb_keypoints):
        keypoints = tf.reshape(parsed_kps, (-1, 2))[:nb_keypoints]
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        # denorm
        x_coords *= img_size[0]
        y_coords *= img_size[1]
        # center
        x_coords -= centroid[0]
        y_coords -= centroid[1]
        # renorm
        x_coords /= (bbox_size // 2)
        y_coords /= (bbox_size // 2)
        return tf.reshape(tf.stack([x_coords, y_coords], axis=1), (nb_keypoints * 2,))

    @staticmethod
    def preprocess_image(image_data, centroid, bbox_size, cropsize):
        """
        Performs preproccessing on a single image for feeding into the network.
        :param image_data: raw image data as a bytestring
        :param centroid: the center of the bounding box to crop to, in pixel coordinates
        :param bbox_size: the side length of the bbox to crop to, in pixels
        :param cropsize: the output size of the cropped image
        :return: the decoded image cropped to a [bbox_size, bbox_size] square centered around centroid
        and resized to [cropsize, cropsize]
        """
        image = tf.io.decode_image(image_data, channels=3)
        # this rescales inputs to the range [-1, 1], which should be what the model expects
        image = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))

        # ensure types
        bbox_size = tf.cast(bbox_size, tf.float32)
        centroid = tf.cast(centroid, tf.float32)

        # convert to [0, 1] relative coordinates
        imdims = tf.cast(tf.shape(image)[:2], tf.float32)
        centroid /= imdims
        bbox_size /= imdims  # will broadcast to shape [2]

        # crop to (bbox_size, bbox_size) centered around centroid and resize to (cropsize, cropsize)
        bbox_size /= 2
        image = tf.squeeze(tf.image.crop_and_resize(
            tf.expand_dims(image, 0),
            [[centroid[0] - bbox_size[0], centroid[1] - bbox_size[1], centroid[0] + bbox_size[0],
              centroid[1] + bbox_size[1]]],
            [0],
            [cropsize, cropsize],
            extrapolation_value=-1
        ))
        image = tf.ensure_shape(image, [cropsize, cropsize, 3])
        return imdims, image
