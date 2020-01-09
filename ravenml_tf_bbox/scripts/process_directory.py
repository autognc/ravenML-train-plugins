import argparse
import numpy as np
import os
import cv2
import ravenml_tf_bbox.validation.utils as utils
import ravenml_tf_bbox.validation.stats as stats
import object_detection.utils.visualization_utils as visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Path to model (.pb)", required=True)
    parser.add_argument('-l', '--labelmap', type=str, help="Path to label map (.pbtxt)", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to image directory", required=True)
    parser.add_argument('-o', '--output', type=str, help="Path to put output", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    parser.add_argument('-v', '--vis', type=bool, help="Write out visualizations (optional)")
    parser.add_argument('--gaussian-noise', type=float, help="Add Gaussian noise with a certain stddev (optional)")
    args = parser.parse_args()

    category_index = utils.get_categories(args.labelmap)
    all_paths = utils.get_image_paths(args.directory)
    image_paths, bbox_paths, metadata_paths = tuple(sorted(paths)[:args.num] for paths in all_paths)
    images = utils.gen_images_from_paths(image_paths)
    all_truths = utils.gen_truth_from_bbox_paths(bbox_paths)
    graph = utils.get_default_graph(args.model)

    if args.gaussian_noise:
        images = (image + np.random.normal(scale=args.gaussian_noise, size=image.shape) for image in images)

    outputs, times = utils.run_inference_for_multiple_images(images, graph)

    all_detections = utils.convert_inference_output_to_detected_objects(category_index, outputs)

    if args.vis:
        print("Writing visualizations...")
        images = utils.gen_images_from_paths(image_paths)  # refresh generator
        if args.gaussian_noise:
            images = (image + np.random.normal(scale=args.gaussian_noise, size=image.shape) for image in images)
        for detections, image, image_path in zip(all_detections, images, image_paths):
            for class_name, detection in detections.items():
                visualization.draw_bounding_box_on_image_array(
                    image, detection.box['ymin'], detection.box['xmin'], detection.box['ymax'], detection.box['xmax'],
                    color='green', thickness=4, display_str_list=[f'{class_name}: {int(detection.score * 100)}%']
                )
            image_name = os.path.basename(image_path)
            cv2.imwrite(os.path.join(args.output, image_name), image)

    confidence, accuracy, recall, precision, iou, parameters =\
        stats.calculate_statistics(all_truths, all_detections, category_index)

    stats.write_stats_to_json(confidence, accuracy, recall, precision, iou, parameters, times, category_index,
                              args.output)


if __name__ == "__main__":
    main()
