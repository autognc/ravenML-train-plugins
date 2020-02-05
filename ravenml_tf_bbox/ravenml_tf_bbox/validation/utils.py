import os
import glob
import tensorflow as tf
import json


def gen_truth_data(dir_path):
    """
    Gets ground truth bboxes and centroids from meta_*.json files.
    :param dir_path: the directory to load metadata from
    :return: a generator that, for each image, yield a tuple (bbox_dict, centroid_dict) where:
        bbox_dict = {classname: bbox} where bbox is a dictionary with keys {xmin, xmax, ymin, ymax}
        and centroid_dict = {classname: centroid} where each centroid = (y, x).
        Both bboxes and centroids are in non-normalized (pixel) coordinates.
    """
    meta_files = sorted(glob.glob(os.path.join(dir_path, "meta_*")))
    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        yield meta['bboxes'], meta['centroids']


def get_image_dataset(dir_path, gaussian_stddev=0.0):
    """
    Get a tf.data.Dataset that yields image files from a directory in sorted order.
    """
    def image_parser(image_path):
        return tf.io.decode_image(tf.io.read_file(image_path), channels=3)

    image_files = sorted(glob.glob(os.path.join(dir_path, "image_*")))
    image_dataset = tf.data.Dataset.from_tensor_slices(image_files).map(image_parser, num_parallel_calls=16)
    if gaussian_stddev > 0:
        image_dataset = image_dataset.map(lambda img: add_gaussian_noise(img, gaussian_stddev), num_parallel_calls=16)
    return image_dataset


def add_gaussian_noise(img, stddev):
    img = tf.cast(img, tf.float32) / 255
    img += tf.random.normal(tf.shape(img), stddev=stddev)
    img = tf.clip_by_value(img, 0, 1)
    return tf.cast(img * 255, tf.uint8)

