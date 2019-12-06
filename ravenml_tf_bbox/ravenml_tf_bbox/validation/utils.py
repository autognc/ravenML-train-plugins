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

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

from ravenml_tf_instance.validation.classes import DetectedClass, TruthClass


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
    mask_paths = []
    metadata_paths = []
    color_paths = []

    for item in os.listdir(dev_path):
        if item.startswith('image'):
            image_paths.append(os.path.join(dev_path, item))

    for impath in image_paths:

        imname = impath.split('/')[-1]
        uid = imname.split('_')[1].split('.')[0]
        mask_name = "mask_{}.png".format(uid)
        meta_name = "meta_{}.json".format(uid)
        color_name = "labels_{}.csv".format(uid)
        
        mask_paths.append(os.path.join(dev_path, mask_name))
        metadata_paths.append(os.path.join(dev_path, meta_name))
        color_paths.append(os.path.join(dev_path, color_name))

    return image_paths, mask_paths, metadata_paths, color_paths


def load_images_from_paths(image_paths):
    images = []
    for impath in image_paths:
        image = Image.open(impath)
        image_np = load_image_into_numpy_array(image)

        images.append(image_np)

    return images


def load_masks_from_paths(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        mask_np = load_image_into_numpy_array(mask)

        masks.append(mask_np)

    return masks

def load_colors_from_paths(color_paths, category_index):
    colors = []


    for color_path in color_paths:
        reader = csv.DictReader(open(color_path))

        temp = {}
        color = {}
        for row in reader:
            key = row['label']
            val = [int(row['R']), int(row['G']), int(row['B'])]
            
            temp[key] = val

        for class_id in category_index:
            name = category_index[class_id]['name']

            if name == "solar_panel":
                color[class_id] = [temp['panel_left'], temp['panel_right']]

            if name in temp:
                color[class_id] = temp[name]

        colors.append(color)

    return colors

def load_centroids_from_paths(metadata_paths, category_index):
    centroids = []

    for meta_path in metadata_paths:
        centroid = {}
        with open(meta_path) as jf:
            data = json.load(jf)

            for class_id in category_index:
                if category_index[class_id]['name'] == 'barrel':
                    centroid[class_id] = tuple(data['truth_centroids']['barrel_center'])
                
                elif category_index[class_id]['name'] == 'solar_panel':
                    left = tuple(data['truth_centroids']['panel_left'])
                    right = tuple(data['truth_centroids']['panel_right'])
                    centroid[class_id] = [left, right]

                else:
                    centroid[class_id] = None

        centroids.append(centroid)

    return centroids


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
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            ostart = time.time()
            if tensor_dict.get('detection_masks') is not None:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, images[0].shape[0], images[0].shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
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
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
 
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

            if score > 0.5:
                if detections.get(class_id) is None:
                    box = output['detection_boxes'][i]
                    mask = output['detection_masks'][i]

                    if class_name == 'solar_panel':
                        detections[class_id] = [DetectedClass(class_id, class_name, round(100 * score, 3), box, mask)]
                    else:
                        detections[class_id] = DetectedClass(class_id, class_name, round(100 * score, 3), box, mask)

                elif class_name == 'solar_panel' and len(detections.get(class_id)) == 1:
                    box = output['detection_boxes'][i]
                    mask = output['detection_masks'][i]
                    detections[class_id].append(DetectedClass(class_id, class_name, round(100 * score, 3), box, mask))

        all_detections.append(detections)


    return all_detections


def detected_visualize_and_save(images, all_detections, output_path, experiment):
    for image_idx in range(0, len(all_detections), 20):

        dirpath = Path(output_path) / 'viz' / 'detected' / str(image_idx)
        os.makedirs(dirpath, exist_ok=True)

        for class_id in all_detections[image_idx]:
            data = all_detections[image_idx][class_id]

            if type(data) is list:
                count = 1
                for d in data:
                    if d.centroid is not None:
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(d.mask)
                        plt.imshow(images[image_idx], cmap='jet', alpha=0.6)
                        plt.scatter(d.centroid[1], d.centroid[0], color='r')
                        plt.text(7, 15, d.class_name + ': ' + str(round(100 * d.score, 2)), c='w')

                        figname = 'panel_{}.png'.format(str(count))
                        figpath = dirpath / figname

                        plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
                        experiment.log_asset(figpath)

                        count += 1

            elif data.centroid is not None:
                fig = plt.figure()
                plt.axis('off')
                plt.imshow(data.mask)
                plt.imshow(images[image_idx], cmap='jet', alpha=0.6)
                plt.scatter(data.centroid[1], data.centroid[0], color='r')
                plt.text(7, 15, data.class_name + ': ' + str(round(100 * data.score, 2)), c='w')

                figname = "_".join(i for i in data.class_name.lower().split(" ")) + '.png'
                figpath = dirpath / figname

                plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
                experiment.log_asset(figpath)

def truth_visualize_and_save(images, all_truths, output_path, experiment):

    for image_idx in range(0, len(all_truths), 20):

        dirpath = Path(output_path) / 'viz' / 'truth' / str(image_idx)
        os.makedirs(dirpath, exist_ok=True)

        for class_id in all_truths[image_idx]:
            data = all_truths[image_idx][class_id]

            if type(data) is list:
                count = 1
                for d in data:
                    if d.centroid is not None:
                        fig = plt.figure()
                        plt.axis('off')
                        plt.imshow(d.mask)
                        plt.imshow(images[image_idx], cmap='jet', alpha=0.6)
                        plt.scatter(d.centroid[1], d.centroid[0], color='r')
                        plt.text(7, 15, d.class_name, c='w')

                        figname = 'panel_{}.png'.format(str(count))
                        figpath = dirpath / figname

                        plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
                        experiment.log_asset(figpath)
                        count += 1  

            elif data.centroid is not None:
                fig = plt.figure()
                plt.axis('off')
                plt.imshow(data.mask)
                plt.imshow(images[image_idx], cmap='jet', alpha=0.6)
                plt.scatter(data.centroid[1], data.centroid[0], color='r')
                plt.text(7, 15, data.class_name, c='w')

                figname = "_".join(i for i in data.class_name.lower().split(" ")) + '.png'
                figpath = dirpath / figname

                plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
                experiment.log_asset(figpath)


def get_truth_masks(masks, colors, centroids, category_index):
    
    all_truths = []

    for mask, color, centroid in zip(masks, colors, centroids):

        # leave until mask colors are fixed in csv
        #color = {1:[191, 195, 206], 2:[193, 195, 1], 3:[198, 0, 9], 4:[0, 199, 24]}
        truths = {}
        for class_id in category_index:
            class_color = color[class_id]
            class_name = category_index[class_id]['name']
            class_cen = centroid[class_id]

            if class_name == 'solar_panel':

                for c, cen in zip(class_color, class_cen):
                    matched = False
                    empty_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                    match = np.where((mask == c).all(axis=2))
                    y, x = match
                    if len(y) != 0 and len(x) != 0:

                        empty_mask[match] = 1
                        matched = True

                    if matched:
                        if truths.get(class_id) is None:
                            truth = TruthClass(class_id, category_index[class_id]['name'], empty_mask, cen)
                            truths[class_id] = [truth]

                        else:
                            truth = TruthClass(class_id, category_index[class_id]['name'], empty_mask, cen)
                            truths[class_id].append(truth)

            
            else:
                matched = False
                empty_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                match = np.where((mask == class_color).all(axis=2))
                y, x = match
                if len(y) != 0 and len(x) != 0:

                    empty_mask[match] = 1
                    matched = True

                if matched:
                    truth = TruthClass(class_id, category_index[class_id]['name'], empty_mask, class_cen)
                    truths[class_id] = truth

        all_truths.append(truths)
    
    return all_truths