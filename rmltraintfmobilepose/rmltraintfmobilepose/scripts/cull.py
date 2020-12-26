"""
Train and Eval CullNet
"""
from scipy.spatial.transform import Rotation
import tensorflow as tf
import numpy as np
import os
import click
import json
import cv2
import tqdm
from .. import utils


class MaskGenerator:
    def __init__(self, size=224, stack=True):
        self.stack = stack
        self.size = size

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        raise NotImplementedError()

    def make_and_apply_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        mask = self.make_binary_mask(image, r_vec, t_vec, focal_length, extra_crop_params)
        cv2.imshow('mask', im2 * 255)
        while True:
            if cv2.waitKey(0) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()
        asdasd


class NumpyMaskProjector(MaskGenerator):
    def __init__(self, all_model_keypoints, dilate_iters=2, **kwargs):
        super().__init__(**kwargs)
        self.dilate_iters = dilate_iters
        self.all_kps_homo = np.hstack([all_model_keypoints, np.ones((len(all_model_keypoints), 1))]).T

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        imdims = np.array(image.shape[:2])
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
        proj = cam_matrix @ np.hstack([rot_matrix, t_vec]) @ self.all_kps_homo
        coords = ((proj / proj[2])[:2].T).astype(np.uint8)
        coords = np.clip(coords, 0, self.size - 1)
        img = np.zeros((self.size, self.size))
        img[coords[:, 1], coords[:, 0]] = 1
        img = cv2.dilate(img, (4, 4), iterations=self.dilate_iters)
        return img


cull_error_metrics = {
    "keypoint_l2": 
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: 
            np.mean([np.linalg.norm(kp_true - kp) for kp, kp_true in zip(kps_pred, kps_true)]),
    "geodesic_rotation":
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true:
            utils.pose.geodesic_error(r_vec, pose_true),
    "position_l2": 
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true:
            utils.pose.position_error(t_vec, position_true)[0]
}


cull_mask_generators = {
    "numpy_stack": lambda *args, **kwargs: NumpyMaskProjector(*args, **kwargs, stack=True),
}


@click.command(help="Trained Pose Model (.h5)")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--keypoints",
    help='Path to 3D reference points .npy file (optional). Defaults to "{directory}/keypoints.npy"',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("-f", "--focal_length", type=float, required=True)
@click.option(
    "-m", 
    "--error_metric", 
    type=click.Choice(cull_error_metrics.keys(), case_sensitive=False), 
    default=next(iter(cull_error_metrics))
)
@click.option(
    "-g", 
    "--mask_mode", 
    type=click.Choice(cull_mask_generators.keys(), case_sensitive=False), 
    default=next(iter(cull_mask_generators))
)
def main(model_path, directory, keypoints, focal_length, error_metric, mask_mode):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(), loss=lambda _: 0,
    )
    error_func = cull_error_metrics[error_metric]
    mask_gen_init = cull_mask_generators[mask_mode]
    if mask_mode.startswith("numpy"):
        # TODO make option
        mask_gen = mask_gen_init(np.load("keypoints700k.npy"), size=224)
    else:
        raise ValueError('Unable to init mask generator')
    X, y = compute_model_error_training_data(model, directory, keypoints, focal_length, error_func, mask_gen)
    print(X.shape, y.shape)


def compute_model_error_training_data(model, directory, keypoints, focal_length, error_func, mask_gen):

    nb_keypoints = model.output.shape[-1] // 2
    cropsize = model.input.shape[1]

    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        directory, cropsize, nb_keypoints=nb_keypoints
    )
    data = data.batch(32)

    inputs = []
    outputs = []

    for image_batch, truth_batch in tqdm.tqdm(data):
        truth_batch = [
            dict(zip(truth_batch.keys(), t)) for t in zip(*truth_batch.values())
        ]
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch = kps_batch * (cropsize // 2) + (cropsize // 2)
        for image, kps, truth in zip(image_batch, kps_batch.numpy(), truth_batch):
            image = tf.cast(image * 127.5 + 127.5, tf.uint8).numpy()
            kps_true = (truth["keypoints"] - truth["centroid"]) / truth[
                "bbox_size"
            ] * cropsize + (cropsize // 2)
            extra_crop_params = {
                "centroid": truth["centroid"],
                "bbox_size": truth["bbox_size"],
                "imdims": truth["imdims"],
            }
            r_vec, t_vec = utils.pose.solve_pose(
                ref_points,
                kps,
                [focal_length, focal_length],
                image.shape[:2],
                extra_crop_params=extra_crop_params,
                ransac=True,
                reduce_mean=False,
            )
            img_mask = mask_gen.make_and_apply_mask(image, r_vec, t_vec, focal_length, extra_crop_params)
            inputs.append(img_mask)
            error = error_func(kps, kps_true, r_vec, t_vec, truth["pose"], truth["position"])
            outputs.append(error)
    return np.array(inputs), np.array(outputs)
