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


def yield_eval_batches(dataset_path, model_path, keypoints_scale, batch_size=32):
    model = tf.keras.models.load_model(model_path)
    cropsize = model.input.shape[1]
    nb_keypoints = model.output.shape[1] // 2
    ref_points = np.load(dataset_path / "keypoints.npy").reshape((-1, 3))[:nb_keypoints]
    test_data = dataset_from_directory(dataset_path / "test", cropsize)
    test_data = test_data.map(
        lambda image, metadata: (
            tf.ensure_shape(image, [cropsize, cropsize, 3]),
            metadata
        )
    )
    test_data = test_data.batch(batch_size)
    for image_batch, metadata in tqdm.tqdm(test_data.as_numpy_iterator()):
        n = len(image_batch)
        pose_batch = metadata['pose']
        bbox_batch = metadata['bboxes']['cygnus']
        kps_pred = model.predict(image_batch)
        kps_actual = np.empty_like(kps_pred)
        for i in range(n):
            bbox = {key: value[i] for key, value in bbox_batch.items()}
            bbox_size = max(bbox['ymax'] - bbox['ymin'], bbox['xmax'] - bbox['xmin']) * 1.25
            centroid = [(bbox['ymax'] + bbox['ymin']) / 2, (bbox['xmax'] + bbox['xmin']) / 2]
            pose = pose_batch[i]
            kps_actual[i] = KeypointsModel.preprocess_keypoints(
                metadata['keypoints'][i].reshape((-1,)), 
                centroid, bbox_size, cropsize, (1024, 1024), 128
            ).numpy()[:nb_keypoints * 2]
        kps = ((kps_pred * (keypoints_scale // 2)) + keypoints_scale // 2).reshape((-1, nb_keypoints, 2))
        kps_truth = ((kps_actual * (keypoints_scale // 2)) + keypoints_scale // 2).reshape((-1, nb_keypoints, 2))
        images = ((image_batch + 1) / 2 * 255).astype(np.uint8)
        yield ref_points, pose_batch, images, kps, kps_truth


def yield_meta_examples(dataset_path, crop_size, batch_size=32):
    test_data = dataset_from_directory(dataset_path / "test", crop_size)
    test_data = test_data.batch(batch_size)
    ref_points = np.load(dataset_path / "keypoints.npy").reshape((-1, 3))
    for image_batch, metadata_batch in tqdm.tqdm(test_data.as_numpy_iterator()):
        images = ((image_batch + 1) / 2 * 255).astype(np.uint8)
        batch_bboxes = metadata_batch['bboxes']['cygnus']
        n = len(image_batch)
        for i in range(n):
            image = images[i]
            bbox = {key: value[i] for key, value in batch_bboxes.items()}
            bbox_size = max(bbox['ymax'] - bbox['ymin'], bbox['xmax'] - bbox['xmin']) * 1.25
            centroid = [(bbox['ymax'] + bbox['ymin']) / 2, (bbox['xmax'] + bbox['xmin']) / 2]
            pose = metadata_batch['pose'][i]
            kps = KeypointsModel.preprocess_keypoints(
                metadata_batch['keypoints'][i].reshape((-1,)), 
                centroid, bbox_size, crop_size, (1024, 1024), 128
            ).numpy().reshape((-1, 2))
            yield image, ref_points, kps, pose


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
