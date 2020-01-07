import tensorflow as tf
import glob
import json
import os
import xml.etree.ElementTree as ET
from .train import PoseRegressionModel


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
        bbox_files = sorted(glob.glob(os.path.join(dir_path, "bboxLabels_*.xml")))
        for image_file, meta_file, bbox_file in zip(image_files, meta_files, bbox_files):
            # load metadata
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            metadata = recursive_map_dict(metadata, tf.convert_to_tensor)

            # load bounding box
            tree = ET.parse(bbox_file)
            bndbox = tree.getroot().find('object').find('bndbox')
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            centroid = tf.convert_to_tensor([(ymax + ymin) / 2, (xmax + xmin) / 2])
            bbox_size = max(xmax - xmin, ymax - ymin)

            # load and crop image
            image_data = tf.io.read_file(image_file)
            image = PoseRegressionModel.preprocess_image(image_data, centroid, bbox_size, cropsize)
            yield image, metadata

    meta_file_0 = glob.glob(os.path.join(dir_path, "meta_*.json"))[0]
    with open(meta_file_0, "r") as f:
        meta0 = json.load(f)
    dtypes = recursive_map_dict(meta0, lambda x: tf.convert_to_tensor(x).dtype)
    return tf.data.Dataset.from_generator(generator, (tf.float32, dtypes))
