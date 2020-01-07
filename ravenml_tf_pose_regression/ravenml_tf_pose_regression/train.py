import tensorflow as tf
import os
import numpy as np
import cv2


class PoseRegressionModel:
    OPTIMIZERS = {
        'SGD': tf.keras.optimizers.SGD,
        'Adam': tf.keras.optimizers.Adam,
        'Adagrad': tf.keras.optimizers.Adagrad,
        'RMSProp': tf.keras.optimizers.RMSprop
    }

    def __init__(self, data_dir, hp):
        """
        :param data_dir: path to data directory
        :param hp: dictionary of hyperparameters
        """
        self.data_dir = data_dir
        self.hp = hp

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
            'pose':
                tf.io.FixedLenFeature([4], tf.float32, default_value=[0.707107, 0.707107, 0.707107, 0.707107])
        }

        cropsize = self.hp['crop_size']

        def _parse_function(example):
            parsed = tf.io.parse_single_example(example, features)
            pose = parsed['pose']

            # find an approximate bounding box to crop the image
            height = tf.cast(parsed['image/height'], tf.float32)
            width = tf.cast(parsed['image/width'], tf.float32)
            xmin = parsed['image/object/bbox/xmin'].values[0] * width
            xmax = parsed['image/object/bbox/xmax'].values[0] * width
            ymin = parsed['image/object/bbox/ymin'].values[0] * height
            ymax = parsed['image/object/bbox/ymax'].values[0] * height
            centroid = tf.stack([(ymax + ymin) / 2, (xmax + xmin) / 2], axis=-1)
            bbox_size = tf.maximum(xmax - xmin, ymax - ymin)

            # random positioning
            if train:
                centroid += tf.random.uniform([2], minval=-bbox_size / 4, maxval=bbox_size / 4)

            # decode, preprocess to [-1, 1] range, and crop image
            image = self.preprocess_image(parsed['image/encoded'], centroid, bbox_size, cropsize)

            # other augmentations
            if train:
                # image values into the [0, 1] format
                image = (image + 1) / 2

                # random multiple of 90 degree rotation
                k = tf.random.uniform([], 0, 4, tf.int32)  # number of CCW 90-deg rotations
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

                # convert back to [-1, 1] format
                image = image * 2 - 1

            #image = (image - self.mean) / self.stdev
            return image, pose

        with open(os.path.join(self.data_dir, f"{split_name}.record.numexamples"), "r") as f:
            num_examples = int(f.read())
        filenames = tf.io.gfile.glob(os.path.join(self.data_dir, f"{split_name}.record-*"))

        return tf.data.TFRecordDataset(filenames, num_parallel_reads=16).map(_parse_function, num_parallel_calls=16), num_examples

    def train(self, logdir):
        train_dataset, num_train = self._get_dataset('train', True)
        train_dataset = train_dataset.shuffle(self.hp['shuffle_buffer']).batch(self.hp['batch_size']).repeat()
        val_dataset, num_val = self._get_dataset('test', False)
        val_dataset = val_dataset.batch(self.hp['batch_size']).repeat()
        """for imb, fb in train_dataset:
            for im, f in zip(imb, fb):
                #im = (im.numpy() * self.stdev + self.mean) * 127.5 + 127.5
                print(f)
                im = im.numpy() * 127.5 + 127.5
                im = im.astype(np.uint8)
                cv2.imshow('a', im)
                cv2.waitKey(0)"""

        # perform training for each training phase
        for i, phase in enumerate(self.hp["phases"]):
            phase_logdir = os.path.join(logdir, f"phase_{i}")
            model_path = os.path.join(phase_logdir, "model.h5")
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
                )
            ]

            # if this is the first phase, generate a new model with fresh weights.
            # otherwise, load the model from the previous phase's best checkpoint
            if i == 0:
                model = self._gen_model()
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

            optimizer = self.OPTIMIZERS[phase['optimizer']](**phase['optimizer_args'])
            model.compile(
                optimizer=optimizer,
                loss=[self.pose_loss],
            )
            model.fit(
                train_dataset,
                epochs=phase['epochs'],
                steps_per_epoch=-(-num_train // self.hp['batch_size']),
                validation_data=val_dataset,
                validation_steps=-(-num_val // self.hp['batch_size']),
                callbacks=callbacks
            )

        return model_path

    def _gen_model(self):
        mobilenet = tf.keras.applications.MobileNetV2(
            include_top=False,
            pooling='avg',
            weights='imagenet',
            input_shape=(self.hp['crop_size'], self.hp['crop_size'], 3)
        )

        feature_map = mobilenet.output
        regression_head_layers = [tf.keras.layers.Flatten()]
        for layer_size in self.hp['regression_head']:
            regression_head_layers += [
                tf.keras.layers.Dense(layer_size, use_bias=True),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(self.hp['dropout'])
            ]
        regression_head_layers += [
            tf.keras.layers.Dense(4),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        ]
        regression_head = tf.keras.Sequential(regression_head_layers, name='regression_head')
        regression_output = regression_head(feature_map)
        return tf.keras.Model(inputs=[mobilenet.input], outputs=[regression_output])

    @staticmethod
    def pose_loss(y_true, y_pred):
        flipped_true = PoseRegressionModel.flip_barrel(y_true)

        dot_a = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=-1)
        dot_b = tf.reduce_sum(tf.multiply(flipped_true, y_pred), axis=-1)
        loss_a = 2 * tf.acos(tf.clip_by_value(dot_a, -1, 1))
        loss_b = 2 * tf.acos(tf.clip_by_value(dot_b, -1, 1))
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

        # convert to [0, 1] relative coordinates
        imdims = tf.cast(tf.shape(image)[:2] - 1, tf.float32)
        centroid /= imdims
        bbox_size /= imdims  # will broadcast to shape [2]

        # crop to (bbox_size, bbox_size) centered around centroid and resize to (cropsize, cropsize)
        bbox_size /= 2
        image = tf.squeeze(tf.image.crop_and_resize(
            tf.expand_dims(image, 0),
            [[centroid[0] - bbox_size[0], centroid[1] - bbox_size[1], centroid[0] + bbox_size[0], centroid[1] + bbox_size[1]]],
            [0],
            [cropsize, cropsize],
            extrapolation_value=-1
        ))
        image = tf.ensure_shape(image, [cropsize, cropsize, 3])
        return image
