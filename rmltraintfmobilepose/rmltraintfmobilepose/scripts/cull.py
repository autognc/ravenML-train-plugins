"""
Train and/or eval CullNet

Show Help
$ ravenml train --config train_config.json tf-mobilepose cull --help

Train a `mobilenetv2_imagenet_mse` cullnet model using `numpy_cut`-masks. Use `cull.npy` to cache error data. 
    `pose_model.h5` is used for generating error data.
$ ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test 
    -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse

Eval model `mobilenetv2_imagenet_mse-1609270326-best.h5` on `~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test` data.
$ ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test 
    -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse -t mobilenetv2_imagenet_mse-1609270326-best.h5
"""
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import moderngl
import click
import glob
import json
import time
import tempfile
import tqdm
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from .. import utils

np.set_printoptions(suppress=True)


class MaskGenerator:
    def __init__(self, size=224, stack=True):
        self.stack = stack
        self.size = size

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, centroid, bbox_size, imdims):
        raise NotImplementedError()

    def make_and_apply_mask(self, image, *args, **kwargs):
        # start = time.time()
        mask = self.make_binary_mask(image, *args, **kwargs)
        # print(time.time() - start)
        assert mask.shape[:2] == image.shape[:2]
        # code for spot-checking masks
        # cv2.imshow("asdf", (image * 127.5 + 127.5).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imshow("asdf", (mask * 255).astype(np.uint8))
        # cv2.waitKey(0)
        if self.stack:
            image_and_mask = np.concatenate([image, mask[..., None]], axis=-1)
            # w, h, c = image.shape
            # assert c == 3
            # image_and_mask = np.empty((w, h, c + 1), dtype=np.float32)
            # image_and_mask[:, :, :3] = image
            # mask is [0, 1]
            # image_and_mask[:, :, 3] = mask * 1
        else:
            image_and_mask = image.copy()
            # -1 dependent on how image is encoded
            image_and_mask[np.where(mask == 0)] = [0, 0, 0]
        return image_and_mask



class OpenGLMaskProjector(MaskGenerator):
    STL_DTYPE = np.dtype(
        [
            ("norm", np.float32, [3]),
            ("vec", np.float32, [3, 3]),
            ("attr", np.uint16, [1]),
        ]
    ).newbyteorder("<")

    VERTEX_SHADER = """
        #version 330
        in vec3 in_vert;
        uniform mat3 projection;
        uniform mat3 rotation;
        uniform vec3 translation;
        uniform int width;
        uniform int height;
        void main() {
            vec3 homog = projection * ((rotation * in_vert) + translation);
            vec2 coord = vec2(homog.x / homog.z, homog.y / homog.z);
            vec2 ndc = vec2(coord.x / width * 2 - 1, coord.y / height * 2 - 1);
            gl_Position = vec4(ndc, 1.0, 1.0);
        }
    """
    FRAGMENT_SHADER = """
        #version 330
        out vec4 color;
        void main() {
            color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        }
    """

    def __init__(self, stl_path, **kwargs):
        super().__init__(**kwargs)

        stl = np.fromfile(stl_path, dtype=self.STL_DTYPE, offset=84)
        verts = stl["vec"].reshape(-1, 3)

        ctx = moderngl.create_context(standalone=True)

        self.prog = ctx.program(
            vertex_shader=self.VERTEX_SHADER,
            fragment_shader=self.FRAGMENT_SHADER,
        )

        self.prog["width"] = self.size
        self.prog["height"] = self.size

        vbo = ctx.buffer(verts.astype("f4").tobytes())
        self.vao = ctx.vertex_array(self.prog, vbo, "in_vert")
        self.fbo = ctx.simple_framebuffer((self.size, self.size))
        self.fbo.use()

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, centroid, bbox_size, imdims):
        origin = centroid - bbox_size / 2
        center = imdims / 2 - origin
        focal_length *= self.size / bbox_size
        center *= self.size / bbox_size
        cam_matrix = np.array(
            [
                [focal_length, 0, center[1]],
                [0, focal_length, center[0]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.prog["projection"] = tuple(cam_matrix.T.flatten())

        self.prog["rotation"] = tuple(
            Rotation.from_rotvec(r_vec.squeeze()).as_matrix().T.flatten()
        )
        self.prog["translation"] = tuple(t_vec.squeeze())
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLES)
        return (
            np.frombuffer(self.fbo.read(), dtype=np.uint8).reshape(
                self.size, self.size, 3
            )[:, :, 0]
        )


def create_model_mobilenetv2_imagenet_mse():
    assert (
        input_shape[-1] == 3
    ), "Using this model requires cut mask generation for 3-channel input data"
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="max",
        weights="imagenet",
        # alpha=0.35,
    )
    new_input = model.input
    feat_out = model.output
    # feat_out = model.get_layer("block_2_add").output
    # feat_out = tf.keras.layers.GlobalMaxPooling2D()(feat_out)
    # feat_out = tf.keras.layers.Dropout(0.5)(feat_out)
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    model = tf.keras.models.Model(new_input, out)
    # model.trainable = True
    # regularizer = tf.keras.regularizers.l2(0.001)
    # for layer in model.layers:
        # for attr in ['kernel_regularizer']:
            # if hasattr(layer, attr):
                # setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    # model_json = model.to_json()

    # Save the weights before reloading the model.
    # tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    # model.save_weights(tmp_weights_path)

    # load the model from the config
    # model = tf.keras.models.model_from_json(model_json)
    
    # Reload the model weights
    # model.load_weights(tmp_weights_path, by_name=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="mse", metrics=[tf.keras.metrics.MeanSquaredError()])
    print(model.summary())
    return model


def create_model_mobilenetv2_fresh_mse():
    weights_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="max", alpha=0.5
    )
    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 4), include_top=False, weights=None, pooling="max", alpha=0.5
    )
    for i in range(3, len(model.layers)):
        model.layers[i].set_weights(weights_model.layers[i].get_weights())
        model.layers[i].trainable = False
    new_input = model.input
    feat_out = model.output
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=[])
    print(full_model.summary())
    del weights_model
    return full_model


# Several possible cullnet models
#   name: model_generator_function
cull_models = {
    "mobilenetv2_imagenet_mse": create_model_mobilenetv2_imagenet_mse,
    "mobilenetv2_fresh_mse": create_model_mobilenetv2_fresh_mse,
}

def conf(kps_pred, kps_true, alpha=2.0, threshold=30):
    dists = np.linalg.norm(kps_pred - kps_true, axis=-1)  # shape (196, 20)
    dists = np.clip(dists, 0, threshold)
    res = (np.exp(alpha * (1 - dists / threshold)) - 1) / (np.exp(alpha) - 1)
    return res


# Several possible error metrics
#   name: (error_calc_function, normalize, denormalize)
cull_error_metrics = {
    "keypoint_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: np.mean(
            conf(kps_pred, kps_true)
        ),
        lambda y: y,
        lambda ynorm: ynorm,
        # lambda y: np.clip((np.log(y) - 5.935) / 0.1731, -3.5, 3.5),
        # lambda ynorm: np.exp(ynorm * 0.1731 + 5.935),
    ),
    "geodesic_rotation": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: utils.pose.geodesic_error(
            r_vec, pose_true
        ),
        lambda y: np.clip((np.log(y) + 3.040) / 1.036, -3.5, 3.5),
        lambda ynorm: np.exp(ynorm * 1.036 + 3.040),
    ),
    "position_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: utils.pose.position_error(
            t_vec, position_true
        ),
        lambda y: y,  # TODO
        lambda ynorm: ynorm,
    ),
}

def reprojection_confidence(r_vec, t_vec, output, cropsize, ref_points):
    unscaled_kps = derive_keypoints(r_vec, t_vec, output["imdims"], output["focal_length"], ref_points)
    kps_reproj = (unscaled_kps - output["centroid"]) / output[
        "bbox_size"
    ] * cropsize + (cropsize // 2)
    # kps_reproj = _project_adjusted(
            # r_vec,
            # t_vec,
            # cropsize,
            # ref_points,
            # output["focal_length"],
            # output["centroid"],
            # output["bbox_size"],
            # output["imdims"]
    # )
    y = np.linalg.norm(kps_reproj - output["kps_true"], axis=-1)
    return np.mean(np.log(1 + y) / 2.5 - 1)
    # return np.mean(conf(kps_reproj, output["kps_true"]))
    


# Several possible mask encodings
#   name: constructor
# "_stack" will stack the mask on to the image as another channel
#       this is what's done in the original cullnet paper
# "_cut" (passed as stack=False) will cut out the shape of the mask from the original image
#       this is cool b/c it allows one to reuse 3-channel pretrained models
cull_mask_generators = {
    "numpy_stack": lambda *args, **kwargs: NumpyMaskProjector(
        *args, **kwargs, stack=True
    ),
    "numpy_cut": lambda *args, **kwargs: NumpyMaskProjector(
        *args, **kwargs, stack=False
    ),
    "opengl_stack": lambda *args, **kwargs: OpenGLMaskProjector(
        *args, **kwargs, stack=True
    ),
    "opengl_cut": lambda *args, **kwargs: OpenGLMaskProjector(
        *args, **kwargs, stack=False
    ),
}


@click.command(help="Train/Use CullNet")
@click.argument("model_path", type=click.Path(exists=True, file_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--keypoints",
    help='Path to 3D reference points .npy file (optional). Defaults to "{directory}/keypoints.npy"',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-f",
    "--focal_length",
    type=float,
    help="Focal length (in pixels). Overrides focal length from truth data. Required if the truth data does not have focal length information.",
)
@click.option(
    "-k",
    "--mask_mode",
    type=click.Choice(cull_mask_generators.keys(), case_sensitive=False),
    default=next(iter(cull_mask_generators)),
)
@click.option(
    "-m",
    "--model_type",
    type=click.Choice(cull_models.keys(), case_sensitive=False),
    default=next(iter(cull_models)),
)
@click.option(
    "-t", "--eval_trained_cull_model", type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "-tt", "--continue_training", type=click.Path(exists=True, dir_okay=False)
)
def main(
    model_path,
    directory,
    keypoints,
    focal_length,
    mask_mode,
    model_type,
    eval_trained_cull_model,
    continue_training,
):

    model = tf.saved_model.load(model_path)
    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))
    nb_keypoints = model.signatures["serving_default"].outputs[0].shape[1]
    ref_points = ref_points[:nb_keypoints]

    mask_gen_init = cull_mask_generators[mask_mode]
    # TODO make hardcoded stuff into options
    if mask_mode.startswith("numpy"):
        mask_gen = mask_gen_init(
            np.load("keypoints700k.npy")
        )
    else:
        mask_gen = mask_gen_init("Cygnus_ENHANCED.stl")

    data_path, outputs = compute_model_error_training_data(
        model, directory, ref_points, focal_length, mask_gen
    )
    files = sorted(glob.glob(os.path.join(data_path, "*.png")))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(lambda fn: tf.cast(tf.io.decode_png(tf.io.read_file(fn), channels=4), tf.float32) / 127.5 - 1)
    # dataset = dataset.map(lambda img: tf.concat([img[..., :3], tf.where(img[..., 3] == -1.0, 0.0, 1.0)[..., None]], axis=-1))

    # for img in dataset:
        # img = (img.numpy() * 127.5 + 127.5).astype(np.uint8)
        # cv2.imshow('asdf', img[..., :3])
        # cv2.waitKey(0)
        # cv2.imshow('asdf', img[..., 3][..., None])
        # cv2.waitKey(0)

    # Train/Test Shuffle & Split
    # these are all normalized

    y = np.array([reprojection_confidence(output["r_vec"], output["t_vec"], output, 224, ref_points) for output in outputs])

    plt.hist(y, bins=100)
    plt.savefig("errors.png")

    if eval_trained_cull_model is None or continue_training is not None:
        np.random.seed(0)
        train_size = int(len(files) * 0.8)
        images_train, outputs_train, images_test, outputs_test = (
                dataset.take(train_size),
                y[:train_size],
                dataset.skip(train_size),
                y[train_size:],
        )
        print("Training cullnet...")
        if continue_training:
            model_name = (
                continue_training.replace("\\", "").replace("/", "").replace(":", "")
            )
            print("Continuing " + model_name)
            cull_model = tf.keras.models.load_model(continue_training, compile=False)
            cull_model.trainable = True
            cull_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="mse")
            cull_model.summary()
        else:
            cull_model = cull_models[model_type]()
        model_name = model_type + "-" + str(int(time.time()))
        cull_model = train_model(
            model_name, cull_model, images_train, outputs_train, images_test, outputs_test, mask_gen, ref_points
        )
    else:
        print("Using pretrained cullnet...", eval_trained_cull_model)
        model_name = (
            eval_trained_cull_model.replace("\\", "").replace("/", "").replace(":", "")
        )
        cull_model = tf.keras.models.load_model(eval_trained_cull_model)



    # print("error        mean=", y.mean(), "std=", y.std())
    # print("error (norm) mean=", ynorm.mean(), "std=", ynorm.std())
    # _, (ax, ax2) = plt.subplots(nrows=2)
    # ax.hist(y)
    # ax.set_title("error")
    # ax2.hist(ynorm)
    # ax2.set_title("error (norm)")
    # plt.savefig("cull-error-dist.png")

    y_pred = cull_model.predict(dataset.batch(64)).squeeze()
    print("Loss: " + str(np.mean((y - y_pred)**2)))

    # Plot & Log results (includes results for all examples and just test examples, normalized and at original scale)
    """
    _, ((ax, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for data, pred_name, axis in zip(
        [
            (ynorm_pred, ynorm),
            (ynorm_test_pred, y_test),
            (error_denorm(ynorm_pred), error_denorm(ynorm)),
            (error_denorm(ynorm_test_pred), error_denorm(y_test)),
        ],
        ["y (norm)", "y_test (norm)", "y", "y_test"],
        [ax, ax2, ax3, ax4],
    ):
        l2_pred = np.linalg.norm(data[0] - data[1])
        corr_pred = np.corrcoef(data[0], data[1])[0, 1]
        print(pred_name, "l2=", l2_pred, "r=", corr_pred)
        axis.scatter(data[0], data[1])
        axis.set_title(pred_name)
    plt.savefig("cull-" + model_name + "-predictions.png")
    """

    plt.clf()
    rot_err = np.array([utils.pose.geodesic_error(output["r_vec"], output["pose"]) for output in outputs])
    pos_err = np.array([utils.pose.position_error(output["t_vec"], output["position"])[1] for output in outputs])
    err = np.cumsum(np.degrees(rot_err)[np.argsort(y_pred)]) / (np.arange(len(rot_err)) + 1)
    x = (np.arange(len(y_pred)) + 1) / len(y_pred)
    fig, ax1 = plt.subplots()
    ax1.scatter(x, err)
    ax1.set_xlabel("proportion of images kept")
    ax1.set_ylabel("mean error", color="blue")
    
    ax2 = ax1.twinx()
    ax2.scatter(x, np.sort(y_pred), color="orange")
    ax2.set_ylabel("confidence threshold", color="orange")
    plt.savefig("er_curve.png")


def train_model(model_name, model, images_train, outputs_train, images_test, outputs_test, mask_gen, ref_points):
    save_name = model_name + "-best.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_name,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=True,
        ),
    ]
    # for layer in model.layers:
        # layer.trainable = True
    try:
        batch_size = 128
        # epochs is set high, use a *single* ^C to finish training
        """
        def gen(train):
            while True:
                cs_dist = []
                if train:
                    idxs = np.random.permutation(len(images_train))
                    images_shuffled = images_train[idxs]
                    outputs_shuffled = outputs_train[idxs]
                    it = zip(images_shuffled, outputs_shuffled)
                else:
                    it = zip(images_test, outputs_test)

                for i, (image, output) in enumerate(it):
                    if i % batch_size == 0:
                        masks, cs = np.empty((batch_size, 224, 224, 4)), np.empty((batch_size,))

                    if train and np.random.rand() < 0.5:
                        r = Rotation.from_euler('xyz', np.random.multivariate_normal(np.zeros(3), np.eye(3) * 20**2), degrees=True)
                        r_vec = (r * Rotation.from_rotvec(output["r_vec"].squeeze())).as_rotvec()
                        t = np.random.multivariate_normal(np.zeros(3), np.diag([0.5, 0.5, 100]))
                        t_vec = output["t_vec"].squeeze() + t
                    else:
                        r_vec = output["r_vec"].squeeze()
                        t_vec = output["t_vec"].squeeze()

                    mask = mask_gen.make_and_apply_mask(image, r_vec, t_vec, output["focal_length"], output["centroid"], output["bbox_size"], output["imdims"])
                    c = reprojection_confidence(r_vec, t_vec, output, mask_gen.size, ref_points)

                    if train:
                        mask = tf.image.random_flip_left_right(mask)
                        mask = tf.image.random_flip_up_down(mask)
                        mask = tf.image.rot90(mask, tf.random.uniform([], 0, 4, tf.int32)).numpy()

                    masks[i % batch_size] = mask
                    cs[i % batch_size] = c
                    cs_dist.append(c)
                    if i == 4000:
                        img = (mask * 127.5 + 127.5).astype(np.uint8)
                        print(c)
                        cv2.imshow('asdf', img[..., :3])
                        cv2.waitKey(0)
                        cv2.imshow('asdf', img[..., 3])
                        cv2.waitKey(0)

                    if i % batch_size == batch_size - 1:
                        yield masks, cs
                plt.clf()
                plt.hist(cs_dist, bins=100)
                plt.savefig("dist.png")
        """

        train_dataset = tf.data.Dataset.zip((images_train, tf.data.Dataset.from_tensor_slices(outputs_train))).cache().shuffle(20000)
        test_dataset = tf.data.Dataset.zip((images_test, tf.data.Dataset.from_tensor_slices(outputs_test)))

        # resampler = tf.data.experimental.rejection_resample(lambda x, y: tf.cast(y > 0, tf.int32), [0.75, 0.25])
        # train_dataset = train_dataset.apply(resampler).map(lambda _, x: x).shuffle(10000)
        # print("plotting...")
        # plt.clf()
        # y = np.array([y.numpy() for _, y in tqdm.tqdm(train_dataset.take(2000))])
        # print(y.mean())
        # plt.hist(y, bins=100)
        # plt.savefig("test.png")
        # train_dataset = tf.data.Dataset.from_generator(gen(True), (tf.float32, tf.float32))
        # test_dataset = tf.data.Dataset.from_generator(gen(False), (tf.float32, tf.float32))
        # rotate = tf.keras.layers.experimental.preprocessing.RandomRotation(1, "constant")
        # translate = tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2, "constant")
        def prep(x, y):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))
            # mask = x != tf.constant([-1.0, -1.0, -1.0])
            z = x[..., :3]
            z = (z + 1) / 2
            z = tf.image.random_brightness(z, 0.2)
            z = tf.clip_by_value(z, 0, 1)
            z = tf.image.random_contrast(z, 0.8, 1.2)
            z = tf.clip_by_value(z, 0, 1)
            z = tf.image.random_saturation(z, 0.7, 1.3)
            z = tf.clip_by_value(z, 0, 1)
            z = z * 2 - 1

            return tf.concat([z, x[..., 3][..., None]], axis=-1), y
            return x, y
            # return tf.where(mask, z, x), y

        train_dataset = train_dataset.map(prep)
        train_dataset = train_dataset.batch(batch_size).prefetch(5)
        test_dataset = test_dataset.batch(batch_size).prefetch(5)

        # imgs, ys = list(train_dataset.take(1).as_numpy_iterator())[0]
        # for img, y in zip(imgs, ys):
            # img = (img * 127.5 + 127.5).astype(np.uint8)
            # print(y)
            # cv2.imshow('asdf', img[..., :3])
            # cv2.waitKey(0)
            # cv2.imshow('asdf', img[..., 3])
            # cv2.waitKey(0)
        model.fit(
            train_dataset,
            epochs=500,
            # steps_per_epoch=len(images_train) // batch_size,
            verbose=1,
            # workers=0,
            validation_data=test_dataset,
            # validation_steps=len(images_test) // batch_size,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("Stopping...")
    print("Loading best save...")
    return tf.keras.models.load_model(save_name)


def derive_keypoints(rot, pos, imdims, focal_length, ref_points):
    r_vec = utils.pose.to_rotation(rot).as_rotvec()
    t_vec = pos
    cam_matrix = np.array(
        [
            [focal_length, 0, imdims[1] // 2],
            [0, focal_length, imdims[0] // 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    kps = cv2.projectPoints(ref_points, r_vec, t_vec, cam_matrix, None)[0][
        :, :, ::-1
    ].squeeze()
    return kps


def preprocess_image(image, cropsize, centroid, bbox_size):
    # convert to [0, 1] relative coordinates
    centroid_norm = centroid / tf.cast(tf.shape(image)[:2], tf.float32)
    bbox_size_norm = bbox_size / tf.cast(tf.shape(image)[:2], tf.float32)  # will broadcast to shape [2]

    # crop to (bbox_size, bbox_size) centered around centroid and resize to (cropsize, cropsize)
    half_bbox_size = bbox_size_norm / 2
    return tf.squeeze(tf.image.crop_and_resize(
            image[None, ...],
            [
                [
                    centroid_norm[0] - half_bbox_size[0],
                    centroid_norm[1] - half_bbox_size[1],
                    centroid_norm[0] + half_bbox_size[0],
                    centroid_norm[1] + half_bbox_size[1],
                ]
            ],
            [0],
            [cropsize, cropsize],
            extrapolation_value=0,
        ))


def compute_model_error_training_data(
    model, directory, ref_points, default_focal_length, mask_gen
):
    cache_path = "cull-cache." + os.path.basename(os.path.normpath(directory))
    if os.path.exists(cache_path):
        print("Using cache...")
        outputs = np.load(os.path.join(cache_path, "outputs.npy"), allow_pickle=True)
        return cache_path, outputs

    print("Using", directory, "to generate cullnet training data...")
    os.makedirs(cache_path)

    nb_keypoints = model.signatures["serving_default"].outputs[0].shape[1]
    cropsize = 224
    ref_points = ref_points[:nb_keypoints]


    image_paths = sorted(glob.glob(os.path.join(directory, "image_*")))
    meta_paths = sorted(glob.glob(os.path.join(directory, "meta_*")))
    inputs = []
    outputs = []
    for i, (image_path, meta_path) in enumerate(tqdm.tqdm(
        list(zip(image_paths, meta_paths))
    )):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # augmentation (train only)
        image = image.astype(np.float32) / 255
        image = tf.image.random_brightness(image, 0.5)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.random_contrast(image, 0.7, 1.4)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.random_saturation(image, 0.6, 1.5)
        image = tf.clip_by_value(image, 0, 1)
        image += tf.random.normal(image.shape, 0.0, 0.05)
        image = tf.clip_by_value(image, 0, 1)
        image = (image.numpy() * 255).astype(np.uint8)

        kps, centroid, bbox_size = model(image)
        kps_uncropped = kps.numpy() / 2 * bbox_size.numpy() + centroid.numpy()
        focal_length = default_focal_length if default_focal_length else meta["focal_length"]
        # get pose solution in uncropped reference frame
        r_vec, t_vec = utils.pose.solve_pose(
            ref_points,
            kps_uncropped,
            [focal_length, focal_length],
            image.shape[:2],
            ransac=True,
            reduce_mean=False,
        )

        cropped = preprocess_image(image, cropsize, centroid, bbox_size).numpy()
        cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
        mask = mask_gen.make_and_apply_mask(cropped, r_vec, t_vec, focal_length, centroid.numpy(), bbox_size.numpy(), np.array(image.shape[:2]))
        cv2.imwrite(os.path.join(cache_path, os.path.basename(image_path).split(".")[0] + ".png"), mask)

        if "keypoints" not in meta:
            unscaled_kps = derive_keypoints(np.array(meta["pose"]), np.array(meta["translation"]), image.shape[:2], focal_length, ref_points)
        else:
            unscaled_kps = np.array(meta["keypoints"])[:nb_keypoints] * image.shape[:2]
        # L2 keypoint error is computed in cropped reference frame
        kps_true = (unscaled_kps - centroid.numpy()) / bbox_size.numpy() * cropsize + (cropsize // 2)
        # kps_reproj = _project_adjusted(r_vec, t_vec, cropsize, ref_points, truth["focal_length"], extra_crop_params)
        outputs.append({
            "kps_true": kps_true,
            "r_vec": r_vec,
            "t_vec": t_vec,
            "pose": np.array(meta["pose"]),
            "position": np.array(meta["translation"]),
            "focal_length": focal_length,
            "centroid": centroid.numpy(),
            "bbox_size": bbox_size.numpy(),
            "imdims": np.array(image.shape[:2]),
        })
        # outputs.append((
            # kps_reproj[:, ::-1], kps_true.numpy(), r_vec, t_vec, truth["pose"].numpy(), truth["position"].numpy()
        # ))
        # print(utils.pose.geodesic_error(r_vec, truth["pose"].numpy()), conf(kps_reproj[:, ::-1], kps_true.numpy()).mean())
        # img = (inputs[-1]* 127.5 + 127.5).astype(np.uint8)
        # for kp in kps_true:
            # cv2.circle(img, tuple(map(int, kp[::-1])), 3, (255, 255, 255), -1)
        # cv2.imshow('adsf', img)
        # cv2.waitKey(0)
    outputs = np.array(outputs)
    np.save(os.path.join(cache_path, "outputs.npy"), outputs)
    return cache_path, outputs
