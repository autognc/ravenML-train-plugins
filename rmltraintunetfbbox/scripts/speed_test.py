import tensorflow as tf
import argparse
import sys
import os
import time
import cv2
import rmltraintunetfbbox.validation.utils as utils
from rmltraintunetfbbox.validation.model import BoundingBoxModel
from rmltraintunetfbbox.validation.stats import BoundingBoxEvaluator
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.exporter import build_detection_graph
from object_detection.builders import model_builder



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="Path to pipeline config file", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to image directory", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    parser.add_argument('--gaussian-noise', type=float, help="Add Gaussian noise with a certain stddev (optional)", default=0.0)
    parser.add_argument('-r', '--rescale', type=float, help="Rescale images by this much (optional)", default=1.0)
    args = parser.parse_args()

    image_dataset = utils.get_image_dataset(args.directory, rescale=args.rescale, gaussian_stddev=args.gaussian_noise)
    truth_data = list(utils.gen_truth_data(args.directory, rescale=args.rescale))
    if args.num:
        image_dataset = image_dataset.take(args.num)
        truth_data = truth_data[:args.num]

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(args.config, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    outputs, placeholder_tensor = build_detection_graph(
        input_type='image_tensor',
        detection_model=detection_model,
        input_shape=None,
        output_collection_name='inference_op',
        graph_hook_fn=None)
    outputs = {'detection_boxes': outputs['detection_boxes'], 'detection_scores': outputs['detection_scores']}

    image_tensor = image_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i, (bbox, centroid) in enumerate(truth_data):
            image = sess.run(image_tensor)
            start = time.time()
            raw_output = sess.run(outputs, feed_dict={placeholder_tensor: image[None, ...]})
            end = time.time()
            print(end - start)


if __name__ == "__main__":
    main()
