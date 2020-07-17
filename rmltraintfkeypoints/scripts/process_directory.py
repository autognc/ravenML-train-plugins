"""
Run inference on a directory of test data using a model in the Tensorflow SavedModel format.
"""

import tensorflow as tf
import numpy as np
import time
import os
import argparse
import json
import cv2
from ravenml.utils.question import user_confirms
from rmltraintfkeypoints import utils, data_utils
from scipy.spatial.transform import Rotation
from rmltraintfkeypoints.train import KeypointsModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Path to saved model")
    parser.add_argument('directory', type=str, help="Path to data directory")
    parser.add_argument('output', type=str, help="Path to write output (json)")
    parser.add_argument('-k', '--keypoints', type=int, help="Number of keypoints", required=True)
    parser.add_argument('-f', '--focal_length', type=float, required=True)
    parser.add_argument('-n', '--num', type=int, help="Number of images to process (optional)")
    args = parser.parse_args()

    if os.path.exists(args.output):
        if not user_confirms('Output path exists. Overwrite?'):
            return

    model = tf.compat.v2.saved_model.load(args.model)
    cropsize = model.__call__.concrete_functions[0].inputs[0].shape[1]
    num_guesses = model.__call__.concrete_functions[0].outputs[0].shape[0]
    data = data_utils.dataset_from_directory(args.directory, cropsize, nb_keypoints=args.keypoints)
    if args.num:
        data = data.take(args.num)

    ref_points = np.load(os.path.join(args.directory, "keypoints.npy")).reshape((-1, 3))[:args.keypoints]
    #ref_points = np.tile(ref_points, [num_guesses // args.keypoints, 1])

    def make_serializable(tensor):
        n = tensor.numpy()
        if isinstance(n, bytes):
            return n.decode('utf-8')
        return n.tolist()

    results = []
    for i, (image, metadata) in enumerate(data):
        image = tf.cast(image * 127.5 + 127.5, tf.uint8).numpy()
        start_time = time.time()
        # kps = (metadata['keypoints'] - metadata['centroid'] + metadata['bbox_size'] / 2) / metadata['bbox_size'] * cropsize
        # for kp in kps.numpy():
            # print(kp)
            # cv2.circle(image, tuple(kp[::-1]), 3, (255, 0, 0))
        # cv2.imshow('asdf', image)
        # cv2.waitKey(0)
        kps = model(image).numpy()
        inference_time = time.time() - start_time
        r_vec, t_vec, cam_matrix, coefs = utils.calculate_pose_vectors(
            ref_points, kps,
            [args.focal_length, args.focal_length], [cropsize, cropsize],
            extra_crop_params={
                'centroid': metadata['centroid'].numpy(),
                'bbox_size': metadata['bbox_size'].numpy(),
                'imdims': metadata['imdims'].numpy(),
            }
        )
        pose = Rotation.from_rotvec(np.squeeze(r_vec))
        pnp_time = time.time() - start_time - inference_time
        print(f"Image: {i}, Model Time: {int(inference_time * 1000)}ms, PnP Time: {int(pnp_time * 1000)}ms")
        result = data_utils.recursive_map_dict(metadata, make_serializable)
        result.update({
            'detected_pose': pose.as_quat()[[3, 0, 1, 2]].tolist(),
            'detected_position': np.squeeze(t_vec).tolist(),
            'inference_time': inference_time,
            'pnp_time': pnp_time,
            'pose_error': utils.geodesic_error(r_vec, metadata['pose'].numpy()).tolist(),
            'position_error': np.linalg.norm(metadata['position'].numpy() - np.squeeze(t_vec))
        })
        results.append(result)

    avg_inference_time = sum(result['inference_time'] for result in results) / len(results)
    avg_pnp_time = sum(result['pnp_time'] for result in results) / len(results)
    avg_pose_error = sum(result['pose_error'] for result in results) / len(results)
    avg_pos_error = sum(result['position_error'] for result in results) / len(results)
    print(f"Average inference time: {int(avg_inference_time* 1000)}ms, average PnP time: {int(avg_pnp_time * 1000)}ms"
          f" average pose error: {avg_pose_error}, average position error: {avg_pos_error}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
