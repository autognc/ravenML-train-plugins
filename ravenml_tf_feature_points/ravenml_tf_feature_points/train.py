import tensorflow as tf
import os
import numpy as np


class FeaturePointsModel:
    DEFAULT_HYPERPARAMETERS = {
        'learning_rate': 0.0045,
        'dropout': 0.5,
        'batch_size': 32,
        'optimizer': 'RMSProp',
        'optimizer_2': 'SGD',
        'momentum': 0.9,
        'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 10,
        'shuffle_buffer': 8192,
        'regression_head_size': 1024,
        'classification_head_size': 256
    }

    OPTIMIZERS = {
        'SGD': lambda hp: tf.keras.optimizers.SGD(learning_rate=hp['learning_rate'], momentum=hp['momentum'], nesterov=hp['nesterov_momentum']),
        'Adam': lambda hp: tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']),
        'Adagrad': lambda hp: tf.keras.optimizers.Adagrad(learning_rate=hp['learning_rate']),
        'RMSProp': lambda hp: tf.keras.optimizers.RMSprop(learning_rate=hp['learning_rate'])
    }

    USED_FEATURE_POINTS = ['panel_left', 'panel_right', 'barrel_top', 'barrel_bottom']
    NUM_USED_FEATURE_POINTS = len(USED_FEATURE_POINTS)

    def __init__(self, data_dir, feature_points, mean, stdev, hp):
        """
        :param data_dir: path to data directory
        :param mean: the mean of the entire dataset
        :param stdev: the standard deviation of the entire dataset
        :param feature_points: list(str) of feature point names, in same order as they appear in the TFRecord
        :param hp: dictionary of hyperparameters, see DEFAULT_HYPERPARAMETERS for available ones
        """
        self.data_dir = data_dir
        self.feature_points = feature_points
        # convert mean and stdev to channel-wise rather than pixel-wise
        self.mean = np.mean(mean, axis=(0, 1))
        self.stdev = np.sqrt(np.mean((mean - self.mean)**2, axis=(0, 1)) + np.mean(stdev**2, axis=(0, 1)))
        # convert mean and stdev to the [-1, 1] format for convenience
        self.mean = (self.mean - 127.5) / 127.5
        self.stdev = self.stdev / (127.5**2)
        self.hp = self.DEFAULT_HYPERPARAMETERS
        self.hp.update(hp)

    def _get_dataset(self, split_name, train):
        """
        Each item in the dataset is a tuple (image, truth_vector).
        image is a Tensor of shape (crop_size, crop_size, 3).
        truth_vector is a Tensor of shape (11,) with elements:
            [
                left_panel_y, right_panel_y, top_barrel_y, bottom_barrel_y, cygnus_logo_y,
                left_panel_x, right_panel_x, top_barrel_x, bottom_barrel_x, cygnus_logo_x,
                logos_facing_camera (0 for false, 1 for true)
            ]

        :param split_name: name of split (e.g. 'train' for 'train-0000-of-0001.tfrecord')
        :param train: boolean, whether or not to perform training augmentations
        :return: a tuple (TFRecordDataset, num_examples) for that split
        """
        features = {
            'height':
                tf.io.FixedLenFeature([], tf.int64),
            'width':
                tf.io.FixedLenFeature([], tf.int64),
            'image_id':
                tf.io.FixedLenFeature([], tf.string),
            'image_data':
                tf.io.FixedLenFeature([], tf.string),
            'image_format':
                tf.io.FixedLenFeature([], tf.string),
            'feature_points':
                tf.io.FixedLenFeature([2, len(self.feature_points)], tf.int64),
            'pose':
                tf.io.FixedLenFeature([4], tf.float32)
        }

        centroid_index = self.feature_points.index("barrel_center")
        top_index = self.feature_points.index("barrel_top")
        bottom_index = self.feature_points.index("barrel_bottom")
        left_index = self.feature_points.index("panel_left")
        right_index = self.feature_points.index("panel_right")
        logo_index = self.feature_points.index("cygnus_logo")
        cropsize = self.hp['crop_size']
        used_feature_point_indices = [self.feature_points.index(s) for s in self.USED_FEATURE_POINTS]

        def _parse_function(example):
            parsed = tf.io.parse_single_example(example, features)
            pose = parsed['pose']
            feature_points = tf.transpose(tf.cast(parsed['feature_points'], tf.float32))

            # find an approximate bounding box to crop the image
            centroid = feature_points[centroid_index]
            barrel_length = tf.norm(feature_points[top_index] - feature_points[bottom_index])
            panel_sep = tf.norm(feature_points[left_index] - feature_points[right_index])
            bbox_size = tf.maximum(barrel_length, panel_sep)

            # random positioning
            if train:
                centroid += tf.random.uniform([2], minval=-bbox_size / 4, maxval=bbox_size / 4)

            # decode, preprocess to [-1, 1] range, and crop image
            image = self.preprocess_image(parsed['image_data'], centroid, bbox_size, cropsize)
            # transform feature points in the same way that the image was cropped
            truth_points = ((feature_points - centroid) / bbox_size + 1) / 2 * cropsize

            # other augmentations
            if train:
                # we have to convert the image values back into the [0, 1] format
                image = (image + 1) / 2

                # random multiple of 90 degree rotation
                k = tf.random.uniform([], 0, 4, tf.int32)  # number of CCW 90-deg rotations
                #cosx = (1 - k % 2) * (-(k % 4) + 1)
                #sinx = (k % 2) * (-(k % 4 - 1) + 1)
                #rot_matrix = tf.convert_to_tensor([[cosx, -sinx], [sinx, cosx]], dtype=tf.float32)
                #center = (cropsize - 1) / 2
                #truth_points = tf.transpose(truth_points)
                #truth_points = tf.matmul(rot_matrix, truth_points - center) + center
                #truth_points = tf.transpose(truth_points)
                #sqrt2 = tf.constant(float(np.sqrt(2)))
                #zero = tf.constant(0.0)
                #one = tf.constant(1.0)
                #two = tf.constant(2.0)
                #three = tf.constant(3.0)
                #w = tf.case([(tf.equal(k, 0), lambda: 1.0), (tf.equal(k, 1), lambda: sqrt2), (tf.equal(k, 2), lambda: 0.0), (tf.equal(k, 3), lambda: -sqrt2)], exclusive=True)
                #z = tf.case([(tf.equal(k, 0), lambda: 0.0), (tf.equal(k, 1), lambda: sqrt2), (tf.equal(k, 2), lambda: 1.0), (tf.equal(k, 3), lambda: sqrt2)], exclusive=True)
                #w = tf.cond(tf.math.equal(k, 0), lambda: one, lambda: tf.cond(k == 1, lambda: sqrt2, lambda: tf.cond(k == 2, lambda: zero, lambda: -sqrt2)))
                #z = tf.cond(tf.math.equal(k, 0), lambda: zero, lambda: tf.cond(k == 2, lambda: one, lambda: sqrt2))
                half_angle = tf.cast(k, tf.float32) * (np.pi / 4)
                w = tf.cos(half_angle)
                z = tf.sin(half_angle)
                wn = w * pose[0] - z * pose[3]
                xn = w * pose[1] - z * pose[2]
                yn = z * pose[1] + w * pose[2]
                zn = w * pose[3] + z * pose[0]
                pose = tf.stack([wn, xn, yn, zn], axis=-1)
                image = tf.image.rot90(image, k)

                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_saturation(image, 0.8, 1.2)
                image = tf.clip_by_value(image, 0, 1)
                # convert back to [-1, 1]
                image = image * 2 - 1

            # cherry-pick the feature points we want
            truth_points = tf.gather(truth_points, used_feature_point_indices)
            # flatten feature points back out to [ycoords] + [xcoords]
            #truth_points = tf.reshape(tf.transpose(truth_points), [self.NUM_USED_FEATURE_POINTS * 2])

            # figure out if the logos are facing the camera
            """theta = tf.asin(2 * (pose[0] * pose[2] - pose[1] * pose[3]))  # the theta Euler angle
            logos_visible = tf.expand_dims(tf.cast(theta < 0, tf.float32), 0)"""  # 1 if they're facing the camera, 0 otherwise

            #image = (image - self.mean) / self.stdev
            return image, pose#(truth_points)#, logos_visible)

        with open(os.path.join(self.data_dir, f"{split_name}.record.numexamples"), "r") as f:
            num_examples = int(f.read())
        filenames = tf.io.gfile.glob(os.path.join(self.data_dir, f"{split_name}.record-*"))

        return tf.data.TFRecordDataset(filenames, num_parallel_reads=16).map(_parse_function, num_parallel_calls=16), num_examples

    def train(self, logdir=None):
        train_dataset, num_train = self._get_dataset('train', True)
        train_dataset = train_dataset.shuffle(self.hp['shuffle_buffer']).batch(self.hp['batch_size']).repeat()
        val_dataset, num_val = self._get_dataset('test', False)
        val_dataset = val_dataset.batch(self.hp['batch_size']).repeat()
        """for imb, fb in train_dataset:
            for im, f in zip(imb, fb):
                #im = (im.numpy() * self.stdev + self.mean) * 127.5 + 127.5
                im = im.numpy() * 127.5 + 127.5
                im = im.astype(np.uint8)
                for feat in f.numpy():
                    cv2.circle(im, tuple(feat[::-1]), 5, (0, 0, 255), -1)
                cv2.imshow('a', im)
                cv2.waitKey(0)"""

        callbacks = []
        if logdir:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False, profile_batch=0, histogram_freq=1))
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                os.path.join(logdir, 'model.h5'),
                monitor='val_pose_loss',
                save_best_only=True,
                mode='min'
            ))
        """callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            min_lr=1e-20,
            patience=5,
            min_delta=0.01,
            mode='min',
            verbose=1
        ))"""
        """callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            mode='min',
            min_delta=1
        ))"""
        model = self._get_model()
        model.summary()
        optimizer = self.OPTIMIZERS[self.hp['optimizer']](self.hp)
        # loss_ratio = (self.hp['crop_size']**2 - 1) / 12
        # binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
        model.compile(
            optimizer=optimizer,
            loss=[self.pose_loss],#, binary_crossentropy_loss],
            loss_weights=[1],#, loss_ratio],
            metrics=[self.pose_loss]
                #[self.average_distance_error] +
                #[self.get_point_average_distance_error(i, u) for i, u in enumerate(self.USED_FEATURE_POINTS)]
            #], ['accuracy']]
        )
        model.fit(
            train_dataset,
            epochs=self.hp['epochs'],
            steps_per_epoch=-(-num_train // self.hp['batch_size']),
            #steps_per_epoch=20,
            validation_data=val_dataset,
            #validation_steps=8,
            validation_steps=-(-num_val // self.hp['batch_size']),
            callbacks=callbacks
        )
        
        model = tf.keras.models.load_model(os.path.join(logdir, 'model.h5'), compile=False)
        start_index = model.layers.index(model.get_layer(self.hp['fine_tune_start']))
        for layer in model.layers[start_index:]:
            layer.trainable = True
        self.hp['learning_rate'] = self.hp['learning_rate_2']
        self.hp['optimizer'] = self.hp['optimizer_2']
        optimizer = self.OPTIMIZERS[self.hp['optimizer']](self.hp)
        model.compile(
            optimizer=optimizer,
            loss=[self.pose_loss],#, binary_crossentropy_loss],
            loss_weights=[1],#, loss_ratio],
            metrics=[self.pose_loss]
                #[self.average_distance_error] +
                #[self.get_point_average_distance_error(i, u) for i, u in enumerate(self.USED_FEATURE_POINTS)]
            #], ['accuracy']]
        )
        model.summary()
        callbacks[0] = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, 'two'), write_graph=False, profile_batch=0, histogram_freq=1)
        callbacks[1] = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, 'two', 'model.h5'),
            monitor='val_pose_loss',
            save_best_only=True,
            mode='min'
        )
        model.fit(
            train_dataset,
            epochs=self.hp['epochs_2'],
            steps_per_epoch=-(-num_train // self.hp['batch_size']),
            validation_data=val_dataset,
            validation_steps=-(-num_val // self.hp['batch_size']),
            callbacks=callbacks
        )

        model = tf.keras.models.load_model(os.path.join(logdir, 'two', 'model.h5'), compile=False)
        start_index = model.layers.index(model.get_layer(self.hp['fine_tune_start_2']))
        for layer in model.layers[start_index:]:
            layer.trainable = True
        model.compile(
            optimizer=optimizer,
            loss=[self.pose_loss],#, binary_crossentropy_loss],
            loss_weights=[1],#, loss_ratio],
            metrics=[self.pose_loss]
                #[self.average_distance_error] +
                #[self.get_point_average_distance_error(i, u) for i, u in enumerate(self.USED_FEATURE_POINTS)]
            #], ['accuracy']]
        )
        model.summary()
        callbacks[0] = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, 'three'), write_graph=False, profile_batch=0, histogram_freq=1)
        callbacks[1] = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(logdir, 'three', 'model.h5'),
            monitor='val_pose_loss',
            save_best_only=True,
            mode='min'
        )
        model.fit(
            train_dataset,
            epochs=self.hp['epochs_2'],
            steps_per_epoch=-(-num_train // self.hp['batch_size']),
            validation_data=val_dataset,
            validation_steps=-(-num_val // self.hp['batch_size']),
            callbacks=callbacks
        )
        return model

    def _get_model(self):
        pooling = self.hp.get('pooling', None)
        mobilenet = tf.keras.applications.MobileNetV2(
            include_top=False,
            pooling=pooling,
            weights='imagenet',
            input_shape=(self.hp['crop_size'], self.hp['crop_size'], 3)
        )
        if self.hp['num_fine_tune_layers'] > 0:
            for layer in mobilenet.layers[:-self.hp['num_fine_tune_layers']]:
                layer.trainable = False
        else:
            mobilenet.trainable = False
        feature_map = mobilenet.output
        regression_head = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.hp['regression_head_size'], use_bias=False),# kernel_regularizer=tf.keras.regularizers.l2(self.hp["l2_reg"])),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(self.hp['dropout']),
            tf.keras.layers.Dense(512, use_bias=False),# kernel_regularizer=tf.keras.regularizers.l2(self.hp["l2_reg"])),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(self.hp['dropout']),
            tf.keras.layers.Dense(4),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
            #tf.keras.layers.Dense(2 * self.NUM_USED_FEATURE_POINTS),
            #tf.keras.layers.Lambda(
                #lambda x: tf.reshape(x, [-1, self.NUM_USED_FEATURE_POINTS, 2]),
            #)
        ], name='feature_points')
        regression_output = regression_head(feature_map)
        """classification_head = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hp['classification_head_size'], activation='relu'),
            tf.keras.layers.Dropout(self.hp['dropout']),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name='logos_facing')"""
        #classification_output = classification_head(model.output)
        # the two outputs will be the feature points vector and the sigmoid probability that the
        # logos are facing the camera
        return tf.keras.Model(inputs=[mobilenet.input], outputs=[regression_output])#, classification_output])

    @staticmethod
    def custom_mse_loss(y_true, y_pred):
        """Same as normal MSE loss, except doesn't care if the panel_left and panel_right are flipped"""
        flipped_pred = FeaturePointsModel.flip_solar_panels(y_pred)
        loss_a = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
        loss_b = tf.reduce_mean(tf.math.squared_difference(y_true, flipped_pred))
        return tf.minimum(loss_a, loss_b)

    @staticmethod
    def average_distance_error(y_true, y_pred):
        flipped_pred = FeaturePointsModel.flip_solar_panels(y_pred)
        error_a = tf.math.reduce_mean(tf.norm(y_true - y_pred, axis=-1))
        error_b = tf.math.reduce_mean(tf.norm(y_true - flipped_pred, axis=-1))
        return tf.minimum(error_a, error_b)

    @staticmethod
    def get_point_average_distance_error(index, name):
        def error(y_true, y_pred):
            if index == 0:
                return tf.minimum(
                    tf.math.reduce_mean(tf.norm(y_true[:, 0, :] - y_pred[:, 0, :], axis=-1)),
                    tf.math.reduce_mean(tf.norm(y_true[:, 0, :] - y_pred[:, 1, :], axis=-1)),
                )
            elif index == 1:
                return tf.minimum(
                    tf.math.reduce_mean(tf.norm(y_true[:, 1, :] - y_pred[:, 0, :], axis=-1)),
                    tf.math.reduce_mean(tf.norm(y_true[:, 1, :] - y_pred[:, 1, :], axis=-1)),
                )
            else:
                return tf.math.reduce_mean(tf.norm(y_true[:, index, :] - y_pred[:, index, :], axis=-1))
        error.__name__ = name
        return error

    @staticmethod
    def flip_solar_panels(vector):
        """
        Given a vector in the format [panel_left_y, panel_right_y, other_y, ..., panel_left_x, panel_right_x, other_x, ...]
        flips the panel_left and panel_right coordinates. Assumes first dimension is batch dimension.
        """
        #coords = tf.reshape(vector, [2, -1])
        indices = tf.concat([[1, 0], tf.range(2, tf.shape(vector)[-2])], axis=0)
        return tf.gather(vector, indices, axis=-2)

    @staticmethod
    def pose_loss(y_true, y_pred):
        flipped_true = FeaturePointsModel.flip_barrel(y_true)

        loss_a = 2 * tf.acos(tf.abs(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)))
        loss_b = 2 * tf.acos(tf.abs(tf.reduce_sum(tf.multiply(flipped_true, y_pred), axis=-1)))
        return tf.minimum(loss_a, loss_b)

    @staticmethod
    def flip_barrel(q):
        """Rotates the barrel 180deg using quaternion magic"""
        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]

        vx = 2 * (y * w + x * z)
        vy = 2 * (y * z - w * x)
        vz = 1 - 2 * (x * x + y * y)

        wn = -(vx * x + vy * y + vz * z)
        xn = vy * z - vz * y + w * vx
        yn = vz * x - vx * z + w * vy
        zn = vx * y - vy * x + w * vz
        return tf.stack([wn, xn, yn, zn], axis=-1)

    @staticmethod
    def preprocess_image(image_data, centroid, bbox_size, cropsize):
        """Centroid and bbox_size in pixels"""
        image = tf.io.decode_image(image_data, channels=3)
        # this rescales inputs to the range [-1, 1], which should be what the model expects
        image = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))
        imdims = tf.cast(tf.shape(image)[:2] - 1, tf.float32)
        # shape (num_features, 2) and in pixel coordinates

        # convert to [0, 1] relative coordinates
        centroid /= imdims
        bbox_size /= imdims  # will broadcast to shape [2]

        # crop to (2 * bbox_size, 2 * bbox_size) centered around centroid and resize to (cropsize, cropsize)
        image = tf.squeeze(tf.image.crop_and_resize(
            tf.expand_dims(image, 0),
            [[centroid[0] - bbox_size[0], centroid[1] - bbox_size[1], centroid[0] + bbox_size[0], centroid[1] + bbox_size[1]]],
            [0],
            [cropsize, cropsize],
            extrapolation_value=-1
        ))
        image = tf.ensure_shape(image, [cropsize, cropsize, 3])
        return image
