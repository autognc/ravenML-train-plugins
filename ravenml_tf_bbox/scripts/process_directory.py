import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import sys
import json
import os
import cv2
import ravenml_tf_bbox.validation.utils as utils
import ravenml_tf_bbox.validation.stats as stats
import object_detection.utils.visualization_utils as visualization
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Path to model (.pb)", required=True)
    parser.add_argument('-l', '--labelmap', type=str, help="Path to label map (.pbtxt)", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to image directory", required=True)
    parser.add_argument('-o', '--output', type=str, help="Path to put output", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    parser.add_argument('-v', '--vis', action="store_true", help="Write out visualizations (optional)")
    parser.add_argument('--gaussian-noise', type=float, help="Add Gaussian noise with a certain stddev (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        print(f"Path '{args.output}' does not exist.")
        sys.exit(1)

    graph = utils.get_model_graph(args.model)
    with graph.as_default():
        image_dataset = utils.get_image_dataset(args.directory)
        bboxes = list(utils.gen_truth_bboxes(args.directory))
        if args.num:
            image_dataset = image_dataset.take(args.num)
            bboxes = bboxes[:args.num]

        if args.gaussian_noise:
            def add_gaussian_noise(img):
                img = tf.cast(img, tf.float32) / 255
                img += tf.random.normal(tf.shape(img), stddev=args.gaussian_noise)
                img = tf.clip_by_value(img, 0, 1)
                return tf.cast(img * 255, tf.uint8)
            image_dataset = image_dataset.map(add_gaussian_noise)

        input_tensor, output_tensors = utils.get_input_and_output_tensors(graph)
        image_tensor = image_dataset.make_one_shot_iterator().get_next()
        category_index = utils.get_category_index(args.labelmap)
        count = 0
        all_outputs = []
        times = []
        with tf.Session() as sess:
            try:
                while True:
                    image = sess.run(image_tensor)
                    start = time.time()
                    raw_output = sess.run(output_tensors, feed_dict={input_tensor: image[None, ...]})
                    end = time.time()
                    output = utils.parse_inference_output(category_index, raw_output, image.shape[0], image.shape[1])
                    if args.vis:
                        for class_name, detections in output.items():
                            score, bbox = detections[0]  # only take top detection b/c there's only once instance
                            visualization.draw_bounding_box_on_image_array(
                                image, bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'],
                                color='green', thickness=4, display_str_list=[f'{class_name}: {int(score * 100)}%'],
                                use_normalized_coordinates=False
                            )
                        cv2.imwrite(os.path.join(args.output, f'{count}.png'), image)
                    print(f'Image: {count}, time: {end - start}')
                    times.append(end - start)
                    all_outputs.append(output)
                    count += 1
            except tf.errors.OutOfRangeError:
                pass

    coco_stats = stats.calculate_coco_statistics(all_outputs, bboxes, category_index)
    avg_time = sum(times[1:]) / (len(times) - 1)  # ignore first time b/c it's always longer
    statistics = {
        'coco_stats': coco_stats,
        'average inference time': avg_time
    }
    with open(os.path.join(args.output, 'stats.json'), 'w') as f:
        json.dump(statistics, f, indent=2)

    stats.plot_pr_curve(all_outputs, bboxes, category_index)
    plt.savefig(os.path.join(args.output, 'pr_curve.png'))


if __name__ == "__main__":
    main()
