import tensorflow as tf
import numpy as np
import traceback
from keras_applications import mobilenet_v2, imagenet_utils
import os
import time
from . import utils
import cv2


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

    def assign_metric(self, y_true, y_pred):
        self.targets.assign(y_true)
        self.outputs.assign(y_pred)
        return 0

    def assign_metric_mobilepose(self, y_true, y_pred):
        self.targets.assign(y_true)
        kps = KeypointsModel.decode_displacement_field(y_pred)  # shape (b, n, 2, h*w)
        kps = tf.transpose(kps, [0, 3, 1, 2])
        kps = tf.reshape(kps, [tf.shape(kps)[0], -1, 2])
        # kps = tf.reduce_min(tf.math.top_k(kps, tf.shape(kps)[0] // 2, sorted=False).values, axis=-1)
        self.outputs.assign(kps)
        return 0

    def calc_pose_error(self):
        label = KeypointsModel.decode_label(self.targets.numpy())
        kps_batch = self.outputs.numpy()
        if len(kps_batch.shape) == 2:
            kps_batch = kps_batch.reshape(kps_batch.shape[0], -1, 2)
        kps_batch = kps_batch * (self.crop_size // 2) + (self.crop_size // 2)

        # image = np.zeros([kps_batch.shape[0], self.crop_size, self.crop_size, 3], dtype=np.uint8)

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

            """for kp_idx in range(0, len(kps), len(self.ref_points)):
                y = int(kps[kp_idx, 0])
                x = int(kps[kp_idx, 1])
                cv2.circle(image[i], (x, y), 3, (0, 255, 255), -1)
            cv2.imshow('asdf', image[i])
            cv2.waitKey(0)"""

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
        self.crop_size_rounded = utils.pow2_round(self.crop_size)
        self.keypoints_mode = 'mask' if (self.hp['model_arch'] == 'unet') else 'coords'

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
            pose = parsed['image/object/pose']
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
                centroid, bbox_size, cropsize, old_dims, self.nb_keypoints, keypoints_mode=self.keypoints_mode)

            # other augmentations
            if train:
                # random multiple of 90 degree rotation
                k = tf.random.uniform([], 0, 4, tf.int32)  # number of CCW 90-deg rotations
                angle = tf.cast(k, tf.float32) * (np.pi / 2)
                # adjust keypoints
                cos = tf.cos(angle)
                sin = tf.sin(angle)
                rot_matrix = tf.convert_to_tensor([[cos, -sin], [sin, cos]])
                keypoints = tf.reshape(keypoints, [-1, 2])[..., None]
                keypoints = tf.reshape(tf.linalg.matmul(rot_matrix, keypoints), [-1])
                # adjust pose
                # TODO I don't think this works, at least not on SPEED
                w = tf.cos(angle / 2)
                z = tf.sin(angle / 2)
                wn = w * pose[0] - z * pose[3]
                xn = w * pose[1] - z * pose[2]
                yn = z * pose[1] + w * pose[2]
                zn = w * pose[3] + z * pose[0]
                pose = tf.stack([wn, xn, yn, zn], axis=-1)
                # adjust image
                image = tf.image.rot90(image, k)


                # image values into the [0, 1] format
                image = (image + 1) / 2

                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
                image = tf.clip_by_value(image, 0, 1)

                # convert back to [-1, 1] format
                image = image * 2 - 1

            truth = self.encode_label(
                keypoints=keypoints,
                pose=pose,
                height=height,
                width=width,
                bbox_size=bbox_size,
                centroid=centroid,
                keypoints_mode=self.keypoints_mode
            )
            return image, truth

        with open(os.path.join(self.data_dir, f"{split_name}.record.numexamples"), "r") as f:
            num_examples = int(f.read())
        filenames = tf.io.gfile.glob(os.path.join(self.data_dir, f"{split_name}.record-*"))

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
        if self.hp['cache_train_data']:
            dataset = dataset.cache()
        if train:
            dataset = dataset.shuffle(self.hp['shuffle_buffer_size'])
        return dataset.map(_parse_function, num_parallel_calls=16), num_examples

    def train(self, logdir, experiment=None):
        train_dataset, num_train = self._get_dataset('train', True)
        train_dataset = train_dataset.batch(self.hp['batch_size']).repeat()
        if self.hp['prefetch_num_batches']:
            train_dataset = train_dataset.prefetch(self.hp['prefetch_num_batches'])
        val_dataset, num_val = self._get_dataset('test', False)
        val_dataset = val_dataset.batch(self.hp['batch_size']).repeat()

        """imgs, kps = list(train_dataset.take(1).as_numpy_iterator())[0]
        for img, kp in zip(imgs, kps):
            kp = self.decode_label(kp)['keypoints'].numpy()
            kp = kp * (self.crop_size / 2) + (self.crop_size / 2)
            img = ((img + 1) / 2 * 255).astype(np.uint8)
            for i in range(len(kp)):
                y = int(kp[i, 0])
                x = int(kp[i, 1])
                cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
            cv2.imshow('test.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)"""

        pose_error_callback = PoseErrorCallback(self.keypoints_3d, self.crop_size, self.hp['pnp_focal_length'], experiment)
        for i, phase in enumerate(self.hp['phases']):
            phase_logdir = os.path.join(logdir, f"phase_{i}")
            model_path = os.path.join(phase_logdir, "model.h5")
            model_path_latest = os.path.join(phase_logdir, "model-latest.h5")
            callbacks = [
                tf.keras.callbacks.TensorBoard(
                    log_dir=phase_logdir,
                    write_graph=False,
                    profile_batch=0
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path_latest,
                    save_best_only=False,
                ),
                # TODO not break w/unet
                pose_error_callback
            ]

            # if this is the first phase, generate a new model with fresh weights.
            # otherwise, load the model from the previous phase's best checkpoint
            if i == 0:
                model = {
                    'mobilenet': self._gen_model_mobilenet,
                    'densenet': self._gen_model_densenet,
                    'unet': self._get_custom_unet,
                    'mobilepose': self._gen_model_mobilepose,
                }[self.hp['model_arch']]()
            else:
                model = tf.keras.models.load_model(
                    os.path.join(logdir, f"phase_{i - 1}", "model.h5"),
                    compile=False
                )

            # allow training on only the layers from start_layer onwards
            start_layer_index = model.layers.index(model.get_layer(phase['start_layer']))
            for layer in model.layers[:start_layer_index]:
                layer.trainable = False
            for layer in model.layers[start_layer_index:]:
                layer.trainable = True
            print(model.summary())

            optimizer = self.OPTIMIZERS[phase['optimizer']](**phase['optimizer_args'])
            if self.hp['model_arch'] == 'mobilepose':
                assign_metric = pose_error_callback.assign_metric_mobilepose
                loss = self.get_mobilepose_loss(model.output_shape[-3:-1])
            else:
                assign_metric = pose_error_callback.assign_metric
                loss = self.make_mse_loss(keypoints_mode=self.keypoints_mode),
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=[assign_metric],
            )
            try:
                model.fit(
                    train_dataset,
                    epochs=phase['epochs'],
                    steps_per_epoch=num_train // self.hp['batch_size'],
                    validation_data=val_dataset,
                    validation_steps=num_val // self.hp['batch_size'],
                    callbacks=callbacks
                )
            except Exception as e:
                print(traceback.format_exc())
                return model_path
            finally:
                if experiment:
                    experiment.log_model(f'phase_{i}', model_path)
                    experiment.log_model(f'phase_{i}', model_path_latest)

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

    def _gen_model_mobilepose(self):
        init_weights = self.hp.get('model_init_weights', '')
        assert init_weights in ['imagenet', '']
        mobilenet = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=init_weights if init_weights != '' else None,
            input_shape=(self.crop_size, self.crop_size, 3),
            pooling=None,
            alpha=1.0
        )
        x = mobilenet.get_layer('block_15_add').output

        # 7x7x160 -> 14x14x96
        x = tf.keras.layers.Conv2DTranspose(
            filters=96,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = tf.keras.layers.ReLU(6.)(x)
        x = tf.keras.layers.concatenate([x, mobilenet.get_layer('block_12_add').output])
        x = mobilenet_v2._inverted_res_block(x, filters=96, alpha=1.0, stride=1,
                                             expansion=6, block_id=17)
        x = mobilenet_v2._inverted_res_block(x, filters=96, alpha=1.0, stride=1,
                                             expansion=6, block_id=18)

        # 14x14x96 -> 28x28x32
        x = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = tf.keras.layers.ReLU(6.)(x)
        x = tf.keras.layers.concatenate([x, mobilenet.get_layer('block_5_add').output])
        x = mobilenet_v2._inverted_res_block(x, filters=32, alpha=1.0, stride=1,
                                             expansion=6, block_id=19)
        x = mobilenet_v2._inverted_res_block(x, filters=32, alpha=1.0, stride=1,
                                             expansion=6, block_id=20)

        # 28x28x32 -> 56x56x24
        x = tf.keras.layers.Conv2DTranspose(
            filters=24,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = tf.keras.layers.ReLU(6.)(x)
        x = tf.keras.layers.concatenate([x, mobilenet.get_layer('block_2_add').output])
        x = mobilenet_v2._inverted_res_block(x, filters=24, alpha=1.0, stride=1,
                                             expansion=6, block_id=21)
        x = mobilenet_v2._inverted_res_block(x, filters=24, alpha=1.0, stride=1,
                                             expansion=6, block_id=22)

        x = tf.keras.layers.SpatialDropout2D(self.hp['dropout'])(x)

        # output 1x1 conv
        x = tf.keras.layers.Conv2D(
            self.nb_keypoints * 2,
            kernel_size=1,
            use_bias=True
        )(x)
        return tf.keras.models.Model(mobilenet.input, x, name='mobilepose')

    def _get_custom_unet(self):
        from keras_unet.models import custom_unet
        assert not self.hp.get('model_init_weights')
        unet_params = self.hp.get('model_unet_params', {})
        unet_model_params = dict(
            input_shape=(self.crop_size_rounded, self.crop_size_rounded, 3),
            use_batch_norm=False,
            num_classes=self.nb_keypoints,
            num_layers=4,
            activation='relu',
            filters=16,
            use_attention=False,
            upsample_mode='deconv',
            dropout=0.3,
            output_activation='linear'
        )
        # Override select model params with the config values.
        overridable_model_params = [
            'use_batch_norm', 'num_layers', 'activation', 
            'filters', 'use_attention', 'upsample_mode'
        ]
        for override_param in overridable_model_params:
            param_val = unet_params.get(override_param)
            if param_val is not None:
                unet_model_params[override_param] = param_val
        unet_model = custom_unet(**unet_model_params)
        # Convert UNet 3d output to keypoints
        unet_in = unet_model.input
        unet_out = unet_model.output
        out_size = self.crop_size_rounded * self.crop_size_rounded * self.nb_keypoints
        x = tf.keras.layers.Reshape((out_size,))(unet_out)
        return tf.keras.models.Model(unet_in, x)

    @staticmethod
    def make_mse_loss(*, keypoints_mode):
        def mse_loss_coords(y_true, y_pred):
            kps = KeypointsModel.decode_label(y_true)['keypoints']
            return tf.keras.losses.mse(tf.reshape(kps, [tf.shape(kps)[0], -1]), y_pred)
        def mse_loss_mask(y_true, y_pred):
            kps = KeypointsModel.decode_label(y_true)['keypoints']
            # TODO decoding seems to be build for coord method
            # so `kps_fixed_shape` is required
            # TODO pass in cropsize and nb_keypoints to compute shape
            kps_fixed_shape = tf.reshape(kps, (-1, 1310720))
            return tf.keras.losses.mse(kps_fixed_shape, y_pred)
        if keypoints_mode == 'coords':
            return mse_loss_coords
        elif keypoints_mode == 'mask':
            return mse_loss_mask
        else:
            raise NotImplementedError(keypoints_mode)

    @staticmethod
    def encode_label(*, keypoints, pose, height, width, bbox_size, centroid, keypoints_mode):
        if len(keypoints.shape) == 3 and keypoints_mode == 'coords':
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
    def get_mobilepose_loss(dfdims):
        def mobilepose_loss(y_true, y_pred):
            kps = KeypointsModel.decode_label(y_true)['keypoints']
            df_true = KeypointsModel.encode_displacement_field(kps, dfdims)
            df_true_flat = tf.reshape(df_true, [tf.shape(df_true)[0], -1])
            df_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
            return tf.keras.losses.mean_absolute_error(df_true_flat, df_pred_flat)
        return mobilepose_loss

    @staticmethod
    def encode_displacement_field(keypoints, dfdims):
        """
        :param keypoints: a shape (b, n, 2) Tensor with N keypoints normalized to (-1, 1)
        :param dfdims: a shape [2] Tensor with the dimensions of the displacement field
        :return: a shape (b, height, width, 2n) Tensor
        """
        delta = 2 / tf.convert_to_tensor(dfdims, dtype=tf.float32)
        y_range = tf.range(-1, 1, delta[0]) + (delta[0] / 2)
        x_range = tf.range(-1, 1, delta[1]) + (delta[1] / 2)
        mgrid = tf.stack(tf.meshgrid(y_range, x_range, indexing='ij'), axis=-1)  # shape (y, x, 2)
        df = keypoints[:, :, None, None, :] - mgrid  # shape (b, n, y, x, 2)
        df = tf.transpose(df, [0, 2, 3, 1, 4])  # shape (b, y, x, n, 2)
        return tf.reshape(df, [tf.shape(keypoints)[0], dfdims[0], dfdims[1], -1])


    @staticmethod
    def decode_displacement_field(df):
        """
        :param df: a shape (b, height, width, 2n) displacement field
        :return: a shape (b, n, 2, height * width) tensor where each keypoint has height * width predictions
        """
        dfdims = tf.shape(df)[1:3]
        df = tf.reshape(df, [tf.shape(df)[0], dfdims[0], dfdims[1], -1, 2])  # shape (b, y, x, n, 2)
        delta = tf.cast(2 / dfdims, tf.float32)
        y_range = tf.range(-1, 1, delta[0]) + (delta[0] / 2)
        x_range = tf.range(-1, 1, delta[1]) + (delta[1] / 2)
        mgrid = tf.stack(tf.meshgrid(y_range, x_range, indexing='ij'), axis=-1)  # shape (y, x, 2)
        keypoints = df + mgrid[:, :, None, :]  # shape (b, y, x, n, 2)
        keypoints = tf.reshape(keypoints, [tf.shape(df)[0], dfdims[0] * dfdims[1], -1, 2])  # shape (b, y*x, n, 2)
        return tf.transpose(keypoints, [0, 2, 3, 1])

    @staticmethod
    def preprocess_keypoints(parsed_kps, centroid, bbox_size, cropsize, img_size, nb_keypoints, keypoints_mode='coords'):
        def _get_image_coords():
            keypoints = tf.reshape(parsed_kps, (-1, 2))[:nb_keypoints]
            x_coords = keypoints[:, 0]
            y_coords = keypoints[:, 1]
            x_coords *= img_size[0]
            y_coords *= img_size[1]
            return x_coords, y_coords
        if keypoints_mode == 'coords':
            keypoints = tf.reshape(parsed_kps, [-1, 2])[:nb_keypoints]
            keypoints = (keypoints - centroid) / (bbox_size / 2)
            return tf.reshape(keypoints, [nb_keypoints * 2])
        elif keypoints_mode == 'mask':
            xc, yc = _get_image_coords()
            xc -= centroid[0] - (bbox_size // 2)
            yc -= centroid[1] - (bbox_size // 2)
            resize_coef = cropsize / bbox_size
            xc *= resize_coef
            yc *= resize_coef
            crop_coords = tf.reshape(tf.stack([xc, yc], axis=1), (nb_keypoints, 2))
            crop_coords_with_idx = tf.concat([
                tf.cast(crop_coords, tf.int64), 
                tf.reshape(tf.range(0, nb_keypoints, 1, dtype=tf.int64), (nb_keypoints, 1))
            ], axis=1)
            mask_sparse = tf.sparse.reorder(tf.sparse.SparseTensor(
                crop_coords_with_idx, 
                tf.ones((nb_keypoints,), dtype=tf.float32), 
                dense_shape=(cropsize, cropsize, nb_keypoints)
            ))
            mask = tf.sparse.to_dense(mask_sparse, default_value=0.)
            # TODO smooth the mask
            return mask
        else:
            raise NotImplementedError(keypoints_mode)

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
