import tensorflow as tf
import numpy as np
import traceback
from tensorflow.python.keras.applications.mobilenet_v2 import _inverted_res_block
import os
from . import utils
import cv2


class PoseErrorCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        ref_points,
        crop_size,
        focal_length,
        object_name,
        real_image_dir=None,
        experiment=None,
    ):
        super().__init__()
        self.model = model
        self.ref_points = ref_points
        self.focal_length = focal_length
        self.experiment = experiment
        self.targets = tf.Variable(0.0, shape=tf.TensorShape(None))
        self.outputs = tf.Variable(0.0, shape=tf.TensorShape(None))
        self.train_errors = []
        self.test_errors = []
        self.comet_step = 0

        self.real_image_dataset = []
        if real_image_dir:
            self.real_image_dataset = utils.data.dataset_from_directory(
                real_image_dir, crop_size, len(ref_points), object_name
            ).batch(32)

    def assign_metric(self, y_true, y_pred):
        self.targets.assign(y_true)
        kps = utils.model.decode_displacement_field(y_pred)
        # kps = tf.reduce_min(tf.math.top_k(kps, tf.shape(kps)[0] // 2, sorted=False).values, axis=-1)
        self.outputs.assign(kps)
        return 0

    def calc_pose_error(self):
        truth_batch = KeypointsModel.decode_label(self.targets.numpy())
        kps_batch = self.outputs.numpy()
        kps_batch_uncropped = (
            kps_batch
            / 2
            * truth_batch["bbox_size"][
                :,
                None,
                None,
                None,
            ]
            + truth_batch["centroid"][:, None, None, :]
        )
        error_batch = np.zeros(kps_batch.shape[0])
        for i, kps in enumerate(kps_batch_uncropped.numpy()):
            r_vec, t_vec = utils.pose.solve_pose(
                self.ref_points,
                kps,
                [self.focal_length, self.focal_length],
                [truth_batch["height"][i], truth_batch["width"][i]],
                ransac=True,
                reduce_mean=False,
            )

            error_batch[i] = utils.pose.geodesic_error(
                r_vec, truth_batch["pose"][i].numpy()
            )

        return error_batch

    def on_epoch_begin(self, epoch, logs=None):
        self.train_errors = []

    def on_epoch_end(self, epoch, logs=None):
        errs_pose = []
        errs_pose_flip = []
        errs_position = []
        for image_batch, truth_batch in self.real_image_dataset:
            kps_batch = self.model.predict(image_batch)
            kps_batch = utils.model.decode_displacement_field(kps_batch)
            kps_batch_uncropped = (
                kps_batch
                / 2
                * truth_batch["bbox_size"][
                    :,
                    None,
                    None,
                    None,
                ]
                + truth_batch["centroid"][:, None, None, :]
            )
            for i, kps in enumerate(kps_batch_uncropped.numpy()):
                r_vec, t_vec = utils.pose.solve_pose(
                    self.ref_points,
                    kps,
                    [truth_batch["focal_length"][i], truth_batch["focal_length"][i]],
                    truth_batch["imdims"][i],
                    ransac=True,
                    reduce_mean=False,
                )
                errs_pose.append(
                    utils.pose.geodesic_error(r_vec, truth_batch["pose"][i])
                )
                errs_pose_flip.append(
                    utils.pose.geodesic_error(r_vec, truth_batch["pose"][i], flip=True)
                )
                errs_position.append(
                    utils.pose.position_error(t_vec, truth_batch["position"][i])[1]
                )
        err_pose = np.degrees(np.mean(errs_pose))
        err_pose_flip = np.degrees(np.mean(errs_pose_flip))
        err_position = np.mean(errs_position)
        print(
            f"\nREAL IMAGE ERROR: {err_pose:.2f} ({err_pose_flip:.2f} flip) deg, {err_position:.2f} pos"
        )
        if self.experiment:
            with self.experiment.validate():
                self.experiment.log_metric("real_image_pose_error_deg", err_pose)
                self.experiment.log_metric(
                    "real_image_pose_error_flip_deg", err_pose_flip
                )
                self.experiment.log_metric("real_image_position_error", err_position)

    def on_train_batch_end(self, batch, logs=None):
        self.comet_step += 1
        mean = np.mean(self.calc_pose_error())
        self.train_errors.append(mean)
        running_mean = np.mean(self.train_errors)
        print(
            f" - pose error: {running_mean:.4f} ({np.degrees(running_mean):.2f} deg)\n",
        )
        if self.experiment:
            self.experiment.log_metric(
                "pose_error_deg", np.degrees(mean), step=self.comet_step
            )

    def on_test_begin(self, logs=None):
        self.test_errors = []

    def on_test_batch_end(self, batch, logs=None):
        self.comet_step += 1
        self.test_errors.append(np.mean(self.calc_pose_error()))

    def on_test_end(self, logs=None):
        mean = np.mean(self.test_errors)
        print(f"Eval pose error: {mean:.4f} ({np.degrees(mean):.2f} deg)", flush=True)
        if self.experiment:
            with self.experiment.validate():
                self.experiment.log_metric(
                    "pose_error_deg", np.degrees(mean), step=self.comet_step
                )


class KeypointsModel:

    OPTIMIZERS = {
        "SGD": tf.keras.optimizers.SGD,
        "Adam": tf.keras.optimizers.Adam,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "RMSProp": tf.keras.optimizers.RMSprop,
    }

    def __init__(self, data_dir, hp, keypoints_3d):
        """
        :param data_dir: path to data directory
        :param hp: dictionary of hyperparameters
        """
        self.data_dir = data_dir
        self.hp = hp
        self.nb_keypoints = hp["keypoints"]
        self.keypoints_3d = keypoints_3d[: self.nb_keypoints]
        self.crop_size = hp["crop_size"]

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
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/object/keypoints": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/object/pose": tf.io.FixedLenFeature([4], tf.float32),
            "image/imageset": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(parsed):
            # find an approximate bounding box to crop the image
            height = tf.cast(parsed["image/height"], tf.float32)
            width = tf.cast(parsed["image/width"], tf.float32)
            pose = parsed["image/object/pose"]
            xmin = parsed["image/object/bbox/xmin"].values[0] * width
            xmax = parsed["image/object/bbox/xmax"].values[0] * width
            ymin = parsed["image/object/bbox/ymin"].values[0] * height
            ymax = parsed["image/object/bbox/ymax"].values[0] * height
            centroid = tf.stack([(ymax + ymin) / 2, (xmax + xmin) / 2], axis=-1)
            bbox_size = tf.maximum(xmax - xmin, ymax - ymin)

            # random positioning
            if train:
                expand_factor = tf.random.uniform(
                    [],
                    minval=self.hp["bbox_expand_min"],
                    maxval=self.hp["bbox_expand_max"],
                )
                # ensures that object does not go off screen
                shift_amount = (expand_factor - 1) * bbox_size / 2
                centroid += tf.random.uniform(
                    [2], minval=-shift_amount, maxval=shift_amount
                )
                bbox_size *= expand_factor
            else:
                bbox_size *= 1.25

            # decode, preprocess to [-1, 1] range, and crop image/keypoints
            old_dims, image = utils.model.preprocess_image(
                parsed["image/encoded"], centroid, bbox_size, self.crop_size
            )

            keypoints = utils.model.preprocess_keypoints(
                parsed["image/object/keypoints"].values,
                centroid,
                bbox_size,
                old_dims,
                self.nb_keypoints,
            )

            # other augmentations
            if train:
                # random multiple of 90 degree rotation
                k = tf.random.uniform(
                    [], 0, 4, tf.int32
                )  # number of CCW 90-deg rotations
                angle = tf.cast(k, tf.float32) * (np.pi / 2)
                # adjust keypoints
                cos = tf.cos(angle)
                sin = tf.sin(angle)
                rot_matrix = tf.convert_to_tensor([[cos, -sin], [sin, cos]])
                keypoints = tf.reshape(keypoints, [-1, 2])[..., None]
                keypoints = tf.reshape(tf.linalg.matmul(rot_matrix, keypoints), [-1])
                # adjust pose
                # TODO this doesn't work with the new CV2-compliant pose reference frame
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

                if "random_hue" in self.hp:
                    image = tf.image.random_hue(image, self.hp["random_hue"])
                    image = tf.clip_by_value(image, 0, 1)

                if "random_brightness" in self.hp:
                    image = tf.image.random_brightness(
                        image, self.hp["random_brightness"]
                    )
                    image = tf.clip_by_value(image, 0, 1)

                if "random_saturation" in self.hp:
                    image = tf.image.random_saturation(
                        image, *self.hp["random_saturation"]
                    )
                    image = tf.clip_by_value(image, 0, 1)

                if "random_contrast" in self.hp:
                    image = tf.image.random_contrast(image, *self.hp["random_contrast"])
                    image = tf.clip_by_value(image, 0, 1)

                if "random_gaussian" in self.hp:
                    if tf.random.uniform([], 0, 1) > 0.5:
                        image += tf.random.normal(
                            tf.shape(image), stddev=self.hp["random_gaussian"]
                        )
                        image = tf.clip_by_value(image, 0, 1)

                if "random_jpeg" in self.hp:
                    image = tf.image.random_jpeg_quality(image, *self.hp["random_jpeg"])
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
            )
            return image, truth

        # with open(
        #     os.path.join(self.data_dir, f"{split_name}.record.numexamples"), "r"
        # ) as f:
        #     num_examples = int(f.read())
        filenames = tf.io.gfile.glob(
            os.path.join(self.data_dir, f"{split_name}.record-*")
        )

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)
        if self.hp["cache_train_data"]:
            dataset = dataset.cache()
        if train:
            dataset = dataset.shuffle(self.hp["shuffle_buffer_size"])

        dataset = dataset.map(
            lambda example: tf.io.parse_single_example(example, features),
            num_parallel_calls=16,
        )

        if self.hp["excluded_imagesets"]:
            dataset = dataset.filter(
                lambda parsed: tf.math.reduce_all(
                    parsed["image/imageset"]
                    != tf.convert_to_tensor(self.hp["excluded_imagesets"])
                )
            )

        dataset = dataset.filter(
            lambda parsed: len(parsed["image/object/bbox/xmin"].values) > 0
        )
        return dataset.map(_parse_function, num_parallel_calls=16)

    def train(self, logdir, experiment=None):
        train_dataset = self._get_dataset("train", True)
        train_dataset = train_dataset.batch(self.hp["batch_size"])
        if self.hp["prefetch_num_batches"]:
            train_dataset = train_dataset.prefetch(self.hp["prefetch_num_batches"])
        val_dataset = self._get_dataset("test", False)
        val_dataset = val_dataset.batch(self.hp["batch_size"])

        # imgs, kps = list(train_dataset.take(1).as_numpy_iterator())[0]
        # for img, kp in zip(imgs, kps):
        #     kp = self.decode_label(kp)["keypoints"].numpy()
        #     kp = kp * (self.crop_size / 2) + (self.crop_size / 2)
        #     img = ((img + 1) / 2 * 255).astype(np.uint8)
        #     for i in range(len(kp)):
        #         y = int(kp[i, 0])
        #         x = int(kp[i, 1])
        #         cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
        #     cv2.imshow("test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)

        for i, phase in enumerate(self.hp["phases"]):
            # if this is the first phase, generate a new model with fresh weights.
            # otherwise, load the model from the previous phase's best checkpoint
            if i == 0:
                model = self._gen_model()
                assert os.path.isfile(self.hp.get("model_checkpoint_path"))
                if self.hp.get("model_checkpoint_path"):
                    model.load_weights(self.hp.get("model_checkpoint_path"))
            else:
                model = tf.keras.models.load_model(
                    os.path.join(logdir, f"phase_{i - 1}", "model.h5"), compile=False
                )

            # allow training on only the layers from start_layer onwards
            start_layer_index = model.layers.index(
                model.get_layer(phase["start_layer"])
            )
            for layer in model.layers[:start_layer_index]:
                layer.trainable = False
            for layer in model.layers[start_layer_index:]:
                layer.trainable = True
            print(model.summary())

            def schedule(epoch):
                curr_stage = next(
                    stage for stage in phase["lr_schedule"] if epoch < stage["epoch"]
                )
                i = phase["lr_schedule"].index(curr_stage)
                prev_epoch = phase["lr_schedule"][i - 1]["epoch"] if i > 0 else 0
                return curr_stage["lr"] * tf.math.exp(
                    tf.cast(curr_stage["exp"] * (prev_epoch - epoch), tf.float32)
                )

            phase_logdir = os.path.join(logdir, f"phase_{i}")
            model_path = os.path.join(phase_logdir, "model.h5")
            model_path_latest = os.path.join(phase_logdir, "model-latest.h5")
            pose_error_callback = PoseErrorCallback(
                model,
                self.keypoints_3d,
                self.crop_size,
                self.hp["pnp_focal_length"],
                object_name=self.hp.get("object_name", "cygnus"),
                real_image_dir=self.hp.get("real_image_dir"),
                experiment=experiment,
            )
            callbacks = [
                tf.keras.callbacks.TensorBoard(
                    log_dir=phase_logdir, write_graph=False, profile_batch=0
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path, monitor="val_loss", save_best_only=True, mode="min"
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    model_path_latest,
                    save_best_only=False,
                ),
                tf.keras.callbacks.LearningRateScheduler(schedule),
                pose_error_callback,
            ]

            optimizer = self.OPTIMIZERS[phase["optimizer"]](**phase["optimizer_args"])
            model.compile(
                optimizer=optimizer,
                loss=self.get_mobilepose_loss(model.output_shape[-3:-1]),
                metrics=[pose_error_callback.assign_metric],
            )
            try:
                model.fit(
                    train_dataset,
                    epochs=phase["lr_schedule"][-1]["epoch"],
                    # steps_per_epoch=num_train // self.hp["batch_size"],
                    validation_data=val_dataset,
                    # validation_steps=num_val // self.hp["batch_size"],
                    callbacks=callbacks,
                )
            except Exception:
                print(traceback.format_exc())
                return model_path
            finally:
                if experiment:
                    experiment.log_model(f"phase_{i}", model_path)
                    experiment.log_model(f"phase_{i}", model_path_latest)

        return model_path

    def _gen_model(self):
        init_weights = self.hp.get("model_init_weights", "")
        assert init_weights in ["imagenet", ""]
        
        mobilenet = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=init_weights,
            input_shape=(self.crop_size, self.crop_size, 3),
            pooling=None,
        )
        
        x = mobilenet.get_layer("block7a_project_bn").output

        # 7x7x160 -> 14x14x112
        x = tf.keras.layers.Conv2DTranspose(
            filters=112, kernel_size=3, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        x = tf.keras.layers.ReLU(6.0)(x)
        x = tf.keras.layers.concatenate([x, mobilenet.get_layer("block5b_add").output])
        x = _inverted_res_block(
            x, filters=112, alpha=1.0, stride=1, expansion=6, block_id=17
        )
        x = _inverted_res_block(
            x, filters=112, alpha=1.0, stride=1, expansion=6, block_id=18
        )

        # 14x14x96 -> 28x28x32
        # x = tf.keras.layers.Conv2DTranspose(
        #     filters=32, kernel_size=3, strides=2, padding="same", use_bias=False
        # )(x)
        # x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        # x = tf.keras.layers.ReLU(6.0)(x)
        # x = tf.keras.layers.concatenate([x, mobilenet.get_layer("block_5_add").output])
        # x = _inverted_res_block(
        #     x, filters=32, alpha=1.0, stride=1, expansion=6, block_id=19
        # )
        # x = _inverted_res_block(
        #     x, filters=32, alpha=1.0, stride=1, expansion=6, block_id=20
        # )
        #
        # # 28x28x32 -> 56x56x24
        # x = tf.keras.layers.Conv2DTranspose(
        #     filters=24, kernel_size=3, strides=2, padding="same", use_bias=False
        # )(x)
        # x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
        # x = tf.keras.layers.ReLU(6.0)(x)
        # x = tf.keras.layers.concatenate([x, mobilenet.get_layer("block_2_add").output])
        # x = _inverted_res_block(
        #     x, filters=24, alpha=1.0, stride=1, expansion=6, block_id=21
        # )
        # x = _inverted_res_block(
        #     x, filters=24, alpha=1.0, stride=1, expansion=6, block_id=22
        # )

        x = tf.keras.layers.SpatialDropout2D(self.hp["dropout"])(x)

        # output 1x1 conv
        x = tf.keras.layers.Conv2D(self.nb_keypoints * 2, kernel_size=1, use_bias=True)(
            x
        )
        return tf.keras.models.Model(mobilenet.input, x, name="mobilepose")

    @staticmethod
    def encode_label(*, keypoints, pose, height, width, bbox_size, centroid):
        if len(keypoints.shape) == 3:
            keypoints = tf.reshape(keypoints, [tf.shape(keypoints)[0], -1])
            return tf.concat(
                (
                    tf.reshape(tf.cast(pose, tf.float32), [-1, 4]),
                    tf.reshape(tf.cast(height, tf.float32), [-1, 1]),
                    tf.reshape(tf.cast(width, tf.float32), [-1, 1]),
                    tf.reshape(tf.cast(bbox_size, tf.float32), [-1, 1]),
                    tf.reshape(tf.cast(centroid, tf.float32), [-1, 2]),
                    tf.cast(keypoints, tf.float32),
                ),
                axis=-1,
            )
        else:
            keypoints = tf.reshape(keypoints, [-1])
            return tf.concat(
                (
                    tf.reshape(tf.cast(pose, tf.float32), [4]),
                    tf.reshape(tf.cast(height, tf.float32), [1]),
                    tf.reshape(tf.cast(width, tf.float32), [1]),
                    tf.reshape(tf.cast(bbox_size, tf.float32), [1]),
                    tf.reshape(tf.cast(centroid, tf.float32), [2]),
                    tf.cast(keypoints, tf.float32),
                ),
                axis=-1,
            )

    @staticmethod
    def decode_label(label):
        if len(label.shape) == 1:
            label = label[None, ...]
        return {
            "pose": tf.squeeze(label[:, :4]),
            "height": tf.squeeze(label[:, 4]),
            "width": tf.squeeze(label[:, 5]),
            "bbox_size": tf.squeeze(label[:, 6]),
            "centroid": tf.squeeze(label[:, 7:9]),
            "keypoints": tf.squeeze(
                tf.reshape(label[:, 9:], [tf.shape(label)[0], -1, 2])
            ),
        }

    @staticmethod
    def get_mobilepose_loss(dfdims):
        def mobilepose_loss(y_true, y_pred):
            kps = KeypointsModel.decode_label(y_true)["keypoints"]
            df_true = utils.model.encode_displacement_field(kps, dfdims)
            df_true_flat = tf.reshape(df_true, [tf.shape(df_true)[0], -1])
            df_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
            return tf.keras.losses.mean_absolute_error(df_true_flat, df_pred_flat)

        return mobilepose_loss
