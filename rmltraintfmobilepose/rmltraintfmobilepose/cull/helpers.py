import moderngl
import numpy as np
import tensorflow as tf
import cv2
from scipy.spatial.transform import Rotation
from scipy.stats import trim_mean
from rmltraintfmobilepose import utils
from ravenml.utils.question import user_confirms, user_input, user_selects
from ravenml.utils.plugins import raise_parameter_error
import os

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
        cv2.imwrite("image.png", (image * 127.5 + 127.5).astype(np.uint8))
        # cv2.waitKey(0)
        cv2.imwrite("mask.png", (mask * 255).astype(np.uint8))
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
    def __init__(self, stl_fn, dilate_iters=2, **kwargs):
        super().__init__(**kwargs)
        self.dilate_iters = dilate_iters
        all_model_keypoints = load_stl(stl_fn)
        self.all_kps_homo = np.hstack(
            [all_model_keypoints, np.ones((len(all_model_keypoints), 1))]
        ).T

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        imdims = np.array(image.shape[:2])
        np.save('out.npy', (imdims, image, r_vec, t_vec, focal_length, extra_crop_params))
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

        verts = load_stl(stl_path)

        ctx = moderngl.create_context(standalone=True, backend="egl")

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


def load_stl(stl_fn):
    stl = np.fromfile(stl_fn, dtype=np.dtype(
        [
            ("norm", np.float32, [3]),
            ("vec", np.float32, [3, 3]),
            ("attr", np.uint16, [1]),
        ]
    ).newbyteorder("<"), offset=84)
    return stl["vec"].reshape(-1, 3)


def create_model_mobilenetv2_imagenet_mse(input_shape, pose_model):
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


def create_model_mobilenetv2_imagenet_mse_low_alpha(input_shape, pose_model):
    assert (
        input_shape[-1] == 3
    ), "Using this model requires cut mask generation for 3-channel input data"
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=True,
        weights="imagenet",
        alpha=0.35,
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


def create_model_mobilenetv2_fresh_mse(input_shape, pose_model):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=True, weights=None
    )
    new_input = model.input
    feat_out = model.layers[-2].output
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[])
    return full_model


def create_model_from_pose(input_shape, pose_model, from_layer="spatial_dropout2d"):
    new_input = pose_model.input
    feat_out = pose_model.get_layer(from_layer).output
    x = feat_out
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[])
    return full_model


# Several possible cullnet models
#   name: model_generator_function
cull_models = {
    "mobilenetv2_imagenet_mse": create_model_mobilenetv2_imagenet_mse,
    "mobilenetv2_imagenet_mse_low_alpha": create_model_mobilenetv2_imagenet_mse_low_alpha,
    "mobilenetv2_fresh_mse": create_model_mobilenetv2_fresh_mse,
    "from_pose": lambda input_shape, pose_model: create_model_from_pose(input_shape, pose_model, "spatial_dropout2d"),
    "from_trunc_pose": lambda input_shape, pose_model: create_model_from_pose(input_shape, pose_model, "block_7_add"),
}


# Several possible error metrics
#   name: (error_calc_function, normalize, denormalize)
cull_error_metrics = {
    "keypoint_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: np.mean(
            np.linalg.norm(kps_pred - kps_true, axis=-1)
        ),
        lambda y: (np.log(y) - 1.7351553) / 1.1415093,
        lambda ynorm: np.exp(ynorm * 1.1415093 + 1.7351553),
    ),
    "keypoint_l2_trimed": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: np.mean(
            np.linalg.norm(trim_mean(kps_pred, 0.1, axis=0) - kps_true)
        ),
        lambda y: (np.log(y) - 3.0916994) / 1.2838193,
        lambda ynorm: np.exp(ynorm * 1.2838193 + 3.0916994),
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


def prepare_for_training(config):
    """ Checks plugin specific config

    Args:
        config (dict): dictionary of plugin specific hyperparameters
    """
    if not config.get("object_name"):
        prompt = "Enter the name of the object of interest:"
        config["object_name"] = user_input(prompt, default="cygnus")
    
    for opt, choices in zip(("mask_mode", "model_type", "error_metric"),
                (cull_mask_generators, cull_models, cull_error_metrics)):
        user_in = config.get(opt)
        if not user_in or user_in not in choices.keys():
            choice = user_selects(f"Select CullNet {opt}", list(choices.keys()))
            config[opt] = choice
    stl_path = config.get("stl_path")
    if not stl_path or not os.path.exists(stl_path):
        stl_path = user_input("Enter Path to STL file", validator= lambda x: os.path.exists(x))
        config["stl_path"] = stl_path
    
    model_path = config.get("model_path")
    if not model_path or not os.path.exists(model_path):
        model_path = user_input("Enter Path to Model .h5 File", validator= lambda x: os.path.exists(x))
        config["model_path"] = model_path
    
    config["batch_size"] = config.get("batch_size", 64)
    config["epochs"] = config.get("epochs", 500)

    return config


