import tensorflow as tf
import argparse
import sys
import os
import cv2
import ravenml_tf_bbox.validation.utils as utils
from ravenml_tf_bbox.validation.model import BoundingBoxModel
from ravenml_tf_bbox.validation.stats import BoundingBoxEvaluator
import object_detection.utils.visualization_utils as visualization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Path to model (.pb)", required=True)
    parser.add_argument('-l', '--labelmap', type=str, help="Path to label map (.pbtxt)", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to image directory", required=True)
    parser.add_argument('-o', '--output', type=str, help="Path to put output", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    parser.add_argument('-v', '--vis', action="store_true", help="Write out visualizations (optional)")
    parser.add_argument('--gaussian-noise', type=float, help="Add Gaussian noise with a certain stddev (optional)", default=0.0)
    parser.add_argument('-r', '--rescale', type=float, help="Rescale images by this much (optional)", default=1.0)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        print(f"Path '{args.output}' does not exist.")
        sys.exit(1)

    image_dataset = utils.get_image_dataset(args.directory, rescale=args.rescale, gaussian_stddev=args.gaussian_noise)
    truth_data = list(utils.gen_truth_data(args.directory, rescale=args.rescale))
    if args.num:
        image_dataset = image_dataset.take(args.num)
        truth_data = truth_data[:args.num]

    model = BoundingBoxModel(args.model, args.labelmap)
    evaluator = BoundingBoxEvaluator(model.category_index)
    image_tensor = image_dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        with model.start_session():
            for i, (bbox, centroid) in enumerate(truth_data):
                image = sess.run(image_tensor)
                if args.vis:
                    output, inference_time, vis_img =\
                        model.run_inference_on_single_image(image, vis=True, vis_threshold=0.0)
                    cv2.imwrite(os.path.join(args.output, f'{str(i).zfill(5)}.png'), vis_img)
                else:
                    output, inference_time = model.run_inference_on_single_image(image)
                evaluator.add_single_result(output, inference_time, bbox, centroid)

    evaluator.dump(os.path.join(args.output, 'results.pickle'))
    evaluator.calculate_default_and_save(args.output)


if __name__ == "__main__":
    main()
