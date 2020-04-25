"""
Run inference on a directory of test data using a model in the Tensorflow SavedModel format.
"""

import tensorflow as tf
import time
import argparse
import json
from rmltraintfposeregression.utils import dataset_from_directory, recursive_map_dict
from rmltraintfposeregression.train import PoseRegressionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Path to saved model", required=True)
    parser.add_argument('-d', '--directory', type=str, help="Path to data directory", required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    args = parser.parse_args()

    model = tf.compat.v2.saved_model.load(args.model)
    cropsize = model.__call__.concrete_functions[0].inputs[0].shape[1]
    data = dataset_from_directory(args.directory, cropsize)
    if args.num:
        data = data.take(args.num)

    def make_serializable(tensor):
        n = tensor.numpy()
        if isinstance(n, bytes):
            return n.decode('utf-8')
        return n.tolist()

    results = []
    for i, (image, metadata) in enumerate(data):
        image = tf.cast(image * 127.5 + 127.5, tf.uint8)
        start_time = time.time()
        detected_pose = model(image)
        time_elapsed = time.time() - start_time
        print("Image: ", i, "Time: ", int(time_elapsed * 1000), "ms")
        result = recursive_map_dict(metadata, make_serializable)
        result.update({
            'detected_pose': make_serializable(detected_pose),
            'time': time_elapsed,
            'pose_error': make_serializable(PoseRegressionModel.pose_loss(metadata['pose'], detected_pose)),
            'centroid': make_serializable(list(metadata['centroids'].values())[0])
        })
        results.append(result)

    avg_time = sum(result['time'] for result in results) / len(results)
    avg_error = sum(result['pose_error'] for result in results) / len(results)
    print(f"Average time: {int(avg_time * 1000)}ms, average error: {avg_error}")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
