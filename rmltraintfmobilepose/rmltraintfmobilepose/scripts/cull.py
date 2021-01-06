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
import time
import tqdm
import cv2
import os

from .. import utils


class MaskGenerator:
    def __init__(self, size=224, stack=True):
        self.stack = stack
        self.size = size

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        raise NotImplementedError()

    def make_and_apply_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        mask = self.make_binary_mask(
            image, r_vec, t_vec, focal_length, extra_crop_params
        )
        assert mask.shape[:2] == image.shape[:2]
        # code for spot-checking masks
        # cv2.imshow("asdf", (image * 127.5 + 127.5).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imshow("asdf", (mask * 255).astype(np.uint8))
        # cv2.waitKey(0)
        if self.stack:
            w, h, c = image.shape
            assert c == 3
            image_and_mask = np.empty((w, h, c + 1))
            image_and_mask[:, :, :3] = image
            # mask is [0, 1]
            image_and_mask[:, :, 3] = mask * 1
        else:
            image_and_mask = image.copy()
            # -1 dependent on how image is encoded
            image_and_mask[np.where(mask == 0)] = [-1, -1, -1]
        return image_and_mask


class NumpyMaskProjector(MaskGenerator):
    def __init__(self, all_model_keypoints, dilate_iters=2, **kwargs):
        super().__init__(**kwargs)
        self.dilate_iters = dilate_iters
        self.all_kps_homo = np.hstack(
            [all_model_keypoints, np.ones((len(all_model_keypoints), 1))]
        ).T

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        imdims = np.array(image.shape[:2])
        coords = _project_adjusted(
            r_vec,
            t_vec,
            self.size,
            imdims,
            self.all_kps_homo,
            focal_length,
            extra_crop_params,
        )
        img = np.zeros((self.size, self.size))
        img[coords[:, 1], coords[:, 0]] = 1
        img = cv2.dilate(img, (4, 4), iterations=self.dilate_iters)
        return img


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

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        origin = (
            np.array(extra_crop_params["centroid"]) - extra_crop_params["bbox_size"] / 2
        )
        center = np.array(extra_crop_params["imdims"]) / 2 - origin
        focal_length *= self.size / extra_crop_params["bbox_size"]
        center *= self.size / extra_crop_params["bbox_size"]
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
        self.prog["translation"] = tuple(t_vec)
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLES)
        return (
            np.frombuffer(self.fbo.read(), dtype=np.uint8).reshape(
                self.size, self.size, 3
            )[:, :, 0]
            / 255
        )


def create_model_mobilenetv2_imagenet_mse(input_shape):
    assert (
        input_shape[-1] == 3
    ), "Using this model requires cut mask generation for 3-channel input data"
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=True,
        weights="imagenet",
        # These are ignored, but required to load weights
        classes=1000,
        classifier_activation="softmax",
    )
    new_input = model.input
    feat_out = model.layers[-2].output
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[])
    return full_model


def create_model_mobilenetv2_fresh_mse(input_shape):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=True, weights=None
    )
    new_input = model.input
    feat_out = model.layers[-2].output
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[])
    return full_model


# Several possible cullnet models
#   name: model_generator_function
cull_models = {
    "mobilenetv2_imagenet_mse": create_model_mobilenetv2_imagenet_mse,
    "mobilenetv2_fresh_mse": create_model_mobilenetv2_fresh_mse,
}


# Several possible error metrics
#   name: (error_calc_function, normalize, denormalize)
cull_error_metrics = {
    "keypoint_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: np.mean(
            np.linalg.norm(kps_pred - kps_true, axis=-1)
        ),
        lambda y: np.clip((np.log(y) - 5.935) / 0.1731, -3.5, 3.5),
        lambda ynorm: np.exp(ynorm * 0.1731 + 5.935),
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
        )[
            0
        ],
        lambda y: y,  # TODO
        lambda ynorm: ynorm,
    ),
}


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
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
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
    "-e",
    "--error_metric",
    type=click.Choice(cull_error_metrics.keys(), case_sensitive=False),
    default=next(iter(cull_error_metrics)),
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
    "-c", "--cache", help="Cache dataset here", type=click.Path(dir_okay=False)
)
def main(
    model_path,
    directory,
    keypoints,
    focal_length,
    error_metric,
    mask_mode,
    model_type,
    eval_trained_cull_model,
    cache,
):

    model = tf.keras.models.load_model(model_path, compile=False)
    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))
    error_func, error_norm, error_denorm = cull_error_metrics[error_metric]

    if not cache or not os.path.exists("cull-masks." + cache):
        print("Using", model_path, "to generate cullnet training data...")
        mask_gen_init = cull_mask_generators[mask_mode]
        # TODO make hardcoded stuff into options
        if mask_mode.startswith("numpy"):
            mask_gen = mask_gen_init(
                np.load("keypoints700k.npy"), size=model.input.shape[1]
            )
        else:
            mask_gen = mask_gen_init("Cygnus_ENHANCED.stl", size=model.input.shape[1])
        X, y = compute_model_error_training_data(
            model, directory, ref_points, focal_length, error_func, mask_gen
        )
        if cache:
            np.save("cull-masks." + cache, X)
            np.save("cull-errors." + cache, y)
    else:
        print("Using cached data...")
        X, y = np.load("cull-masks." + cache), np.load("cull-errors." + cache)
    print("Loaded dataset: ", X.shape, " -> ", y.shape)

    # Ensure error data/labels looks good to feed into the model
    ynorm = error_norm(y)
    print("error        mean=", y.mean(), "std=", y.std())
    print("error (norm) mean=", ynorm.mean(), "std=", ynorm.std())
    _, (ax, ax2) = plt.subplots(nrows=2)
    ax.hist(y)
    ax.set_title("error")
    ax2.hist(ynorm)
    ax2.set_title("error (norm)")
    plt.savefig("cull-error-dist.png")

    # Train/Test Shuffle & Split
    np.random.seed(0)
    shuffle_idxs = np.random.permutation(len(X))
    train_size = int(len(X) * 0.7)
    train_idxs, test_idxs = shuffle_idxs[:train_size], shuffle_idxs[train_size:]
    # these are all normalized
    X_train, y_train, X_test, y_test = (
        X[train_idxs],
        ynorm[train_idxs],
        X[test_idxs],
        ynorm[test_idxs],
    )

    if eval_trained_cull_model is None:
        print("Training cullnet...", X_train.shape, X_test.shape)
        model_name = model_type + "-" + str(int(time.time()))
        cull_model = cull_models[model_type](X.shape[1:])
        cull_model = train_model(
            model_name, cull_model, X_train, y_train, X_test, y_test
        )
    else:
        print("Using pretrained cullnet...", eval_trained_cull_model)
        model_name = (
            eval_trained_cull_model.replace("\\", "").replace("/", "").replace(":", "")
        )
        cull_model = tf.keras.models.load_model(eval_trained_cull_model)

    ynorm_pred = cull_model.predict(X).squeeze()
    ynorm_test_pred = cull_model.predict(X_test).squeeze()

    # Plot & Log results (includes results for all examples and just test examples, normalized and at original scale)
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


def train_model(model_name, model, X_train, y_train, X_test, y_test):
    save_name = model_name + "-best.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_name,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=True,
        )
    ]
    try:
        # epochs is set high, use a *single* ^C to finish training
        model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=500,
            verbose=2,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("Stopping...")
    print("Loading best save...")
    return tf.keras.models.load_model(save_name)


def derive_keypoints(truth, ref_points):
    r_vec = utils.pose.to_rotation(truth["pose"]).as_rotvec()
    t_vec = truth["position"].numpy()
    imdims = truth["imdims"]
    focal_length = truth["focal_length"]
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


def _project_adjusted(
    r_vec, t_vec, size, imdims, all_kps_homo, focal_length, extra_crop_params
):
    original_imdims = np.array(extra_crop_params["imdims"])
    origin = (
        np.array(extra_crop_params["centroid"]) - extra_crop_params["bbox_size"] / 2
    )
    center = original_imdims / 2 - origin
    focal_length *= imdims / extra_crop_params["bbox_size"]
    center *= imdims / extra_crop_params["bbox_size"]
    cam_matrix = np.array(
        [[focal_length[1], 0, center[1]], [0, focal_length[0], center[0]], [0, 0, 1]],
        dtype=np.float32,
    )
    rot_matrix = Rotation.from_rotvec(r_vec.reshape((3,))).as_matrix()
    proj = cam_matrix @ np.hstack([rot_matrix, t_vec]) @ all_kps_homo
    coords = ((proj / proj[2])[:2].T).astype(np.uint8)
    return np.clip(coords, 0, size - 1)


def compute_model_error_training_data(
    model, directory, ref_points, default_focal_length, error_func, mask_gen
):
    nb_keypoints = model.output.shape[-1] // 2
    cropsize = model.input.shape[1]
    ref_points = ref_points[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        directory,
        cropsize,
        nb_keypoints=nb_keypoints,
        focal_length=default_focal_length,
    )
    data = data.batch(32)

    inputs = []
    outputs = []

    for image_batch, truth_batch in tqdm.tqdm(data):
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch_cropped = kps_batch * (cropsize // 2) + (cropsize // 2)
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
        truth_batch = [
            dict(zip(truth_batch.keys(), t)) for t in zip(*truth_batch.values())
        ]
        for image, kps_cropped, kps_uncropped, truth in zip(
            image_batch.numpy(),
            kps_batch_cropped.numpy(),
            kps_batch_uncropped.numpy(),
            truth_batch,
        ):
            # get pose solution in uncropped reference frame
            r_vec, t_vec = utils.pose.solve_pose(
                ref_points,
                kps_uncropped,
                [truth["focal_length"], truth["focal_length"]],
                truth["imdims"],
                ransac=True,
                reduce_mean=False,
            )

            extra_crop_params = {
                "centroid": truth["centroid"],
                "bbox_size": truth["bbox_size"],
                "imdims": truth["imdims"],
            }
            img_mask = mask_gen.make_and_apply_mask(
                image, r_vec, t_vec, truth["focal_length"], extra_crop_params
            )
            inputs.append(img_mask)

            if "keypoints" not in truth:
                unscaled_kps = derive_keypoints(truth, ref_points)
            else:
                unscaled_kps = truth["keypoints"]
            # L2 keypoint error is computed in cropped reference frame
            kps_true = (unscaled_kps - truth["centroid"]) / truth[
                "bbox_size"
            ] * cropsize + (cropsize // 2)
            error = error_func(
                kps_cropped, kps_true, r_vec, t_vec, truth["pose"], truth["position"]
            )
            outputs.append(error)
    return np.array(inputs), np.array(outputs)
