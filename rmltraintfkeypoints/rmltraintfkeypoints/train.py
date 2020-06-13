import tensorflow as tf
import numpy as np
import os


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
        self.keypoints_3d = keypoints_3d
        self.nb_keypoints = hp['keypoints']

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

        cropsize = self.hp['crop_size']

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

            return image, keypoints

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
            )
        ]

        model = self._gen_model()
        print(model.summary())

        optimizer = self.OPTIMIZERS[self.hp['optimizer']](**self.hp['optimizer_args'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
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

    def _gen_model(self):
        # TODO support more options
        assert self.hp['model_arch'] == 'densenet'
        assert self.hp['model_init_weights'] == 'imagenet'

        keras_app_args = dict(
            include_top=False, weights='imagenet', 
            input_shape=(self.hp['crop_size'], self.hp['crop_size'], 3), 
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

    @staticmethod
    def preprocess_keypoints(parsed_kps, centroid, bbox_size, cropsize, img_size, nb_keypoints):
        tf.ensure_shape(parsed_kps, (256,))
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
        imdims = tf.cast(tf.shape(image)[:2] - 1, tf.float32)
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
