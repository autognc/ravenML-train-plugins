from .train import KeypointsModel
from scipy.spatial.transform import Rotation
import tensorflow as tf
import numpy as np
import tqdm
import glob
import json
import cv2
import os


def recursive_map_dict(d, f):
    if isinstance(d, dict):
        return {k: recursive_map_dict(v, f) for k, v in d.items()}
    return f(d)


def dataset_from_directory(dir_path, cropsize):
    """
    Get a Tensorflow dataset that generates samples from a directory with test data
    that is not in TFRecord format (i.e. a directory with image_*.png, meta_*.json, and bboxLabels_*.xml files).
    The images are cropped to the spacecraft using the bounding box truth data in the XML files.
    :param dir_path: the path to the directory
    :param cropsize: the output size for the images, in pixels
    :return: a Tensorflow dataset that generates (image, metadata) tuples where image is a [cropsize, cropsize, 3]
    Tensor and metadata is a nested dictionary of Tensors.
    """
    def generator():
        image_files = sorted(glob.glob(os.path.join(dir_path, "image_*")))
        meta_files = sorted(glob.glob(os.path.join(dir_path, "meta_*.json")))
        for image_file, meta_file in zip(image_files, meta_files):
            # load metadata
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            metadata = recursive_map_dict(metadata, tf.convert_to_tensor)

            # load bounding box
            bbox = metadata['bboxes']['cygnus']
            xmin = bbox['xmin']
            xmax = bbox['xmax']
            ymin = bbox['ymin']
            ymax = bbox['ymax']
            centroid = tf.convert_to_tensor([(ymax + ymin) / 2, (xmax + xmin) / 2], dtype=tf.float32)
            bbox_size = tf.cast(tf.maximum(xmax - xmin, ymax - ymin), tf.float32) * 1.25

            # load and crop image
            image_data = tf.io.read_file(image_file)
            _, image = KeypointsModel.preprocess_image(image_data, centroid, bbox_size, cropsize)
            yield image, metadata

    meta_file_0 = glob.glob(os.path.join(dir_path, "meta_*.json"))[0]
    with open(meta_file_0, "r") as f:
        meta0 = json.load(f)
    dtypes = recursive_map_dict(meta0, lambda x: tf.convert_to_tensor(x).dtype)
    return tf.data.Dataset.from_generator(generator, (tf.float32, dtypes))


def yield_eval_batches(dataset_path, model_path, pnp_crop_size, batch_size=32):
    model = tf.keras.models.load_model(model_path)
    cropsize = model.input.shape[1]
    nb_keypoints = model.output.shape[1] // 2
    ref_points = np.load(dataset_path / "keypoints.npy").reshape((-1, 3))[:nb_keypoints]
    test_data = dataset_from_directory(dataset_path / "test", cropsize)
    test_data = test_data.map(
        lambda image, metadata: (
            tf.ensure_shape(image, [cropsize, cropsize, 3]),
            tf.ensure_shape(metadata["pose"], [4])
        )
    )
    test_data = test_data.batch(batch_size)
    for image_batch, pose_batch in tqdm.tqdm(test_data.as_numpy_iterator()):
        images = ((image_batch + 1) / 2 * 255).astype(np.uint8)
        kps_pred = model.predict(image_batch)
        kps = ((kps_pred * (pnp_crop_size // 2)) + pnp_crop_size // 2).reshape((-1, nb_keypoints, 2))
        yield ref_points, pose_batch, images, kps


def calculate_pose_vectors(referance_points, keypoints, focal_length, image_size):
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    cam_matrix = np.array([
        [focal_length, 0, image_size // 2],
        [0, focal_length, image_size // 2],
        [0, 0, 1]
    ], dtype=np.float32)
    ret, r_vec, t_vec = cv2.solvePnP(
        referance_points, keypoints,
        cam_matrix, dist_coeffs)
    assert ret, 'Pose solve failed.'
    return r_vec, t_vec, cam_matrix, dist_coeffs


def rvec_geodesic_error(r_vec, pose_quat):
    r_pred = Rotation.from_rotvec(r_vec.squeeze())
    w, x, y, z = pose_quat
    r_truth = Rotation.from_quat([x, y, z, w])
    return min(
        geodesic_error(_adj_rotation_z(r_pred, 90), r_truth), 
        geodesic_error(_adj_rotation_z(r_pred, -90), r_truth))


def _adj_rotation_z(rot, degs):
    x, y, z = rot.as_euler('xyz', degrees=True)
    return Rotation.from_euler('xyz', [x, y, z + degs], degrees=True)


def geodesic_error(r1, r2):
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    dot = np.abs(np.sum(q1 * q2))
    return 2 * np.arccos(np.clip(dot, 0, 1))
