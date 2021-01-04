import tensorflow as tf
import argparse
import sys
import os
import cv2
import time
from PIL import Image
import shutil
import subprocess
import rmltraintfbbox.validation.utils as utils
from rmltraintfbbox.validation.stats import BoundingBoxEvaluator
import object_detection.utils.visualization_utils as visualization
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder
from object_detection import model_lib_v2, model_lib
import yaml


def mkdir(path):
    try:
        os.makedirs(path)
    except Exception:
        pass

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

def process_directory(exportdir, 
                    labelmap, 
                    directory, 
                    outputdir,
                    eval_name, 
                    num=None, 
                    vis=False, 
                    gaussian_noise=0.0, 
                    rescale=1.0):
    
    mkdir(outputdir)
    image_dataset = utils.get_image_dataset(directory, rescale=rescale, gaussian_stddev=gaussian_noise)
    truth_data = list(utils.gen_truth_data(directory, rescale=rescale))

    if num:
        image_dataset = image_dataset.take(num)
        truth_data = truth_data[:num]

    detection_model = tf.saved_model.load(exportdir + '/saved_model')

    category_index = get_category_index(labelmap)
    evaluator = BoundingBoxEvaluator(category_index)
    for (i, (bbox, centroid, z)), image in zip(enumerate(truth_data), image_dataset):
        true_shape = tf.expand_dims(tf.convert_to_tensor(image.shape), axis=0)
        start = time.time()
        output = detection_model(tf.cast(tf.expand_dims(image, axis=0), dtype=tf.uint8))
        inference_time = time.time() - start
        output['detection_classes'] = output['detection_classes'] - 1
        evaluator.add_single_result(output, true_shape, inference_time, bbox, centroid)
        if vis:
            drawn_img = visualization.draw_bounding_boxes_on_image_tensors(tf.cast(tf.expand_dims(image, axis=0), dtype=tf.uint8), 
                                        output['detection_boxes'], tf.cast(output['detection_classes'] + 1, dtype=tf.int32), 
                                        output['detection_scores'], category_index, max_boxes_to_draw=1, min_score_thresh=0, 
                                        use_normalized_coordinates=True)
            tf.keras.preprocessing.image.save_img(output+f'/img{i}.png', drawn_img[0])

    evaluator.dump(os.path.join(outputdir, 'validation_results.pickle'))
    evaluator.calculate_default_and_save(outputdir)

def prep_model(base_dir, bucket, uuid):
    try: 
        model_path = os.path.join(base_dir, uuid, 'export.zip')
        s3_uri = 's3://' + bucket + '/extras/' + uuid + '/export.zip'                           
        subprocess.call(["aws", "s3", "cp", s3_uri, str(model_path), '--quiet'])
        shutil.unpack_archive(model_path, os.path.join(base_dir, uuid))
        return os.path.join(base_dir, uuid, 'export')
    except:
        return False

def get_imageset(base_dir, bucket, prefix):
    try:
        s3_uri = 's3://' + bucket + '/' + prefix                           
        test_path = os.path.join(base_dir, prefix)
        subprocess.call(["aws", "s3", "sync", s3_uri, str(test_path), '--quiet'])
        return test_path
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="Path to config.yml", required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    base_dir = config['base_dir']
    mkdir(base_dir)
    imageset_bucket = config['imageset_bucket']
    model_bucket = config['model_bucket']
    models = config['models']
    imagesets = config['imagesets']
    labelmap = config['labelmap']
    viz = config.get('viz')
    imagesets = [ imgset for imgset in imagesets if get_imageset(base_dir, imageset_bucket, imgset)]
    for model in models:
        exportdir = prep_model(base_dir, model_bucket, model)
        if not exportdir:
            continue
        for imgset in imagesets:
            outputdir = os.path.join(base_dir, model, imgset)
            process_directory(exportdir, 
                            labelmap, 
                            os.path.join(base_dir, imgset), 
                            outputdir,
                            imgset, 
                            vis=viz)
                                
            
if __name__ == "__main__":
    main()
