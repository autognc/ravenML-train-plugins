import tensorflow as tf
import argparse
import sys
import os
import cv2
import time
from PIL import Image
import rmltraintfbbox.validation.utils as utils
from rmltraintfbbox.validation.stats import BoundingBoxEvaluator
import object_detection.utils.visualization_utils as visualization
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
from object_detection import model_lib_v2, model_lib

def get_category_index(label_path: str):
    label_map = label_map_util.load_labelmap(label_path)
    num_classes = get_num_classes(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def get_num_classes(label_path):
    with open(label_path, "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)
    return num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exportdir', type=str, help="Path to export directory", required=True)
    parser.add_argument('-l', '--labelmap', type=str, help="Path to label map (.pbtxt)", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to image directory", required=True)
    parser.add_argument('-o', '--output', type=str, help="Path to put output", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    parser.add_argument('-v', '--vis', action="store_true", help="Write out visualizations (optional)")
    parser.add_argument('--gaussian-noise', type=float, help="Add Gaussian noise with a certain stddev (optional)", default=0.0)
    parser.add_argument('-r', '--rescale', type=float, help="Rescale images by this much (optional)", default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    image_dataset = utils.get_image_dataset(args.directory, rescale=args.rescale, gaussian_stddev=args.gaussian_noise)
    truth_data = list(utils.gen_truth_data(args.directory, rescale=args.rescale))

    if args.num:
        image_dataset = image_dataset.take(args.num)
        truth_data = truth_data[:args.num]

    detection_model = tf.saved_model.load(args.exportdir + '/saved_model')

    category_index = get_category_index(args.labelmap)
    evaluator = BoundingBoxEvaluator(category_index)
    for (i, (bbox, centroid, z)), image in zip(enumerate(truth_data), image_dataset):
        true_shape = tf.expand_dims(tf.convert_to_tensor(image.shape), axis=0)
        start = time.time()
        output = detection_model.call(tf.expand_dims(image, axis=0))
        inference_time = time.time() - start
        output['detection_classes'] = output['detection_classes'] - 1
        evaluator.add_single_result(output, true_shape, inference_time, bbox, centroid)
        if args.vis:
            drawn_img = visualization.draw_bounding_boxes_on_image_tensors(tf.cast(tf.expand_dims(image, axis=0), dtype=tf.uint8), 
                                        output['detection_boxes'], tf.cast(output['detection_classes'] + 1, dtype=tf.int32), 
                                        output['detection_scores'], category_index, max_boxes_to_draw=1, min_score_thresh=0, 
                                        use_normalized_coordinates=True)
            tf.keras.preprocessing.image.save_img(args.output+f'/img{i}.png', drawn_img[0])

    evaluator.dump(os.path.join(args.output, 'validation_results.pickle'))
    evaluator.calculate_default_and_save(args.output)

if __name__ == "__main__":
    main()
