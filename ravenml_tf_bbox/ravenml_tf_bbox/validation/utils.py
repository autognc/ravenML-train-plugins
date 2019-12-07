import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path
import itertools
import csv
import json

import numpy as np
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from ravenml_tf_bbox.validation.classes import DetectedClass, TruthClass


def get_num_classes(label_path):
    with open(label_path, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)

    return num_classes


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    image = image.convert('RGB')
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_categories(label_path: str):
    label_map = label_map_util.load_labelmap(label_path)
    num_classes = get_num_classes(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)

    category_index = label_map_util.create_category_index(categories)


    return category_index


def get_image_paths(dev_path):
    image_paths = []
    bbox_paths = []
    metadata_paths = []

    for item in os.listdir(dev_path):
        if item.startswith('image'):
            image_paths.append(os.path.join(dev_path, item))

    for impath in image_paths:

        imname = impath.split('/')[-1]
        uid = imname.split('_')[1].split('.')[0]
        bbox_name = "bboxLabels_{}.xml".format(uid)
        meta_name = "meta_{}.json".format(uid)
        
        bbox_paths.append(os.path.join(dev_path, bbox_name))
        metadata_paths.append(os.path.join(dev_path, meta_name))

    return image_paths, bbox_paths, metadata_paths

def gen_images_from_paths(image_paths):
    for impath in image_paths:
        image = Image.open(impath)
        image_np = load_image_into_numpy_array(image)

        yield image_np

def gen_truth_from_bbox_paths(bbox_paths):

    all_truths = []

    for bboxpath in bbox_paths:

        truth = {}

        tree = ET.parse(bboxpath)
        root = tree.getroot()
        size = root.find("size")
        xdim = int(size.find("width").text)
        ydim = int(size.find("height").text)

        for member in root.findall('object'):
            name = member.find("name").text

            bndbox = member.find("bndbox")

            box = {}
            box['xmin'] = int(bndbox.find("xmin").text)
            box['xmax'] = int(bndbox.find("xmax").text)
            box['ymin'] = int(bndbox.find("ymin").text)
            box['ymax'] = int(bndbox.find("ymax").text)

            t = TruthClass(name, box, xdim, ydim)

            # only handles one detection per class per image
            truth[name] = t

        all_truths.append(truth)

    return all_truths


def get_default_graph(model_path):

    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def run_inference_for_multiple_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []
            dict_time = []
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            ostart = time.time()
            if tensor_dict.get('detection_boxes') is not None:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            count = 1
            for image in images:
                # Run inference
                start = time.time()
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
                end = time.time()
                #print('inference time : {}'.format(end - start))
 
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
 
                output_dict_array.append(output_dict)
                dict_time.append(end - start)

                if count % 25 == 0:
                    print("{} images done".format(str(count)))

                count += 1


    return output_dict_array, dict_time


def convert_inference_output_to_detected_objects(category_index, outputs):
    all_detections = []
    for output in outputs:

        detections = {}
        for i in range(len(output['detection_scores'])):
            score = output['detection_scores'][i]
            class_id = output['detection_classes'][i]
            class_name = category_index[class_id]['name']

            if detections.get(class_name) is None:
                det_box = output['detection_boxes'][i]
                box = {
                    'xmin': det_box[1],
                    'xmax': det_box[3],
                    'ymin': det_box[0],
                    'ymax': det_box[2]
                }
                detections[class_name] = DetectedClass(class_name, score, box)

        all_detections.append(detections)

    return all_detections