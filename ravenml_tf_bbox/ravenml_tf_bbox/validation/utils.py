import warnings
warnings.filterwarnings("ignore")
import os
import time
import glob
from collections import defaultdict
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import label_map_util


def gen_truth_bboxes(dir_path):
    """
    Gets ground truth bboxes from bboxLabels_*.xml files.
    :param dir_path: the directory to load bboxes from
    :yield: a dict of the form {classname: [bbox]} where bboxes are dicts
    with keys xmin, xmax, ymin, ymax (non-normalized).
    """
    bbox_files = sorted(glob.glob(os.path.join(dir_path, "bboxLabels_*.xml")))
    for bbox_file in bbox_files:
        # load bounding boxes
        truth = defaultdict(list)
        tree = ET.parse(bbox_file)
        root = tree.getroot()
        size = root.find("size")
        # xdim = int(size.find("width").text)
        # ydim = int(size.find("height").text)
        for classmember in root.findall('object'):
            classname = classmember.find("name").text
            for bndbox in classmember.findall("bndbox"):
                bbox = {
                    'xmin': int(bndbox.find("xmin").text),
                    'xmax': int(bndbox.find("xmax").text),
                    'ymin': int(bndbox.find("ymin").text),
                    'ymax': int(bndbox.find("ymax").text)
                }
                truth[classname].append(bbox)
        yield truth


def get_image_dataset(dir_path):
    """
    Get a tf.data.Dataset that yields image files from a directory in sorted order.
    """
    def image_parser(image_path):
        return tf.io.decode_image(tf.io.read_file(image_path), channels=3)

    image_files = sorted(glob.glob(os.path.join(dir_path, "image_*")))
    image_dataset = tf.data.Dataset.from_tensor_slices(image_files).map(image_parser, num_parallel_calls=16)
    return image_dataset


def get_num_classes(label_path):
    with open(label_path, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)
    return num_classes


def get_category_index(label_path: str):
    label_map = label_map_util.load_labelmap(label_path)
    num_classes = get_num_classes(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def get_model_graph(model_path):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def get_input_and_output_tensors(graph):
    """
    Returns (image_tensor, output_tensors) where image_tensor is a placeholder to be used in feed_dict and
    output_tensors is a dictionary of fetchable output tensors including num_detections, detection_boxes,
    detection_scores, and detection_classes.
    """
    output_tensors = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes']:
        tensor_name = key + ':0'
        output_tensors[key] = graph.get_tensor_by_name(tensor_name)
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    return image_tensor, output_tensors


def parse_inference_output(category_index, output, image_height, image_width):
    """
    Parses the raw output of the object detection model into a more sensible format.
    :param category_index: a category index created with `get_category_index`
    :param output: a dict obtained by running the model and fetching, at the very least, all of the output tensors
    provided by `get_input_and_output_tensors`.
    :return: a dictionary of the form {classname: [(confidence, bbox)]} where bbox is a
    dict with keys xmin, xmax, ymin, ymax (non-normalized).
    """
    # unpack the outputs, which come with a batch dimension
    num_detections = int(output['num_detections'][0])
    detection_classes = output['detection_classes'][0].astype(np.int)
    detection_boxes = output['detection_boxes'][0]
    detection_scores = output['detection_scores'][0]

    detections = defaultdict(list)
    for i in range(num_detections):
        score = detection_scores[i]
        box = detection_boxes[i]
        class_id = detection_classes[i]
        class_name = category_index[class_id]['name']
        bbox = {
            'xmin': box[1] * image_width,
            'xmax': box[3] * image_width,
            'ymin': box[0] * image_height,
            'ymax': box[2] * image_height
        }
        detections[class_name].append((score, bbox))

    return detections

