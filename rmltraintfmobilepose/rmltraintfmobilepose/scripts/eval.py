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
from scipy.spatial.transform import Rotation
from .. import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to saved model")
    parser.add_argument("directory", type=str, help="Path to data directory")
    parser.add_argument("output", type=str, help="Path to write output (json)")
    parser.add_argument(
        "-k", "--keypoints", type=int, help="Number of keypoints", required=True
    )
    parser.add_argument("-f", "--focal_length", type=float, required=True)
    parser.add_argument(
        "-n", "--num", type=int, help="Number of images to process (optional)"
    )
    parser.add_argument("--render", type=str, help="Path to store keypoint renders")
    args = parser.parse_args()

    if os.path.exists(args.output):
        if not user_confirms("Output path exists. Overwrite?"):
            return

    model = tf.compat.v2.saved_model.load(args.model)
    cropsize = model.__call__.concrete_functions[0].inputs[0].shape[1]
    data = utils.data.dataset_from_directory(
        args.directory, cropsize, nb_keypoints=args.keypoints
    )
    if args.num:
        data = data.take(args.num)

    ref_points = np.load(os.path.join(args.directory, "keypoints.npy")).reshape(
        (-1, 3)
    )[: args.keypoints]
    # ref_points = np.tile(ref_points, [num_guesses // args.keypoints, 1])

    def make_serializable(tensor):
        n = tensor.numpy()
        if isinstance(n, bytes):
            return n.decode("utf-8")
        return n.tolist()

    results = []
    for i, (image, metadata) in enumerate(data):
        image = tf.cast(image * 127.5 + 127.5, tf.uint8).numpy()
        start_time = time.time()
        kps = model(image).numpy()
        inference_time = time.time() - start_time
        r_vec, t_vec, cam_matrix, coefs = utils.pose.solve_pose(
            ref_points,
            kps,
            [args.focal_length, args.focal_length],
            [cropsize, cropsize],
            extra_crop_params={
                "centroid": metadata["centroid"].numpy(),
                "bbox_size": metadata["bbox_size"].numpy(),
                "imdims": metadata["imdims"].numpy(),
            },
            ransac=True,
        )
        pose = Rotation.from_rotvec(np.squeeze(r_vec))
        pnp_time = time.time() - start_time - inference_time
        print(
            f"Image: {i}, Model Time: {int(inference_time * 1000)}ms, PnP Time: {int(pnp_time * 1000)}ms"
        )
        result = utils.data.recursive_map_dict(metadata, make_serializable)
        pos_err, pos_err_norm = utils.pose.position_error(
            metadata["position"].numpy(), t_vec
        )
        result.update(
            {
                "detected_pose": pose.as_quat()[[3, 0, 1, 2]].tolist(),
                "detected_position": np.squeeze(t_vec).tolist(),
                "inference_time": inference_time,
                "pnp_time": pnp_time,
                "pose_error": utils.pose.geodesic_error(
                    r_vec, metadata["pose"].numpy()
                ).tolist(),
                "position_error": pos_err.tolist(),
                "position_error_norm": pos_err_norm.tolist(),
            }
        )
        results.append(result)

        if args.render:
            os.makedirs(args.render, exist_ok=True)
            kps = kps.reshape(-1, args.keypoints, 2).transpose([1, 0, 2])
            hues = np.linspace(
                0, 360, num=args.keypoints, endpoint=False, dtype=np.float32
            )
            colors = np.stack(
                [
                    hues,
                    np.ones(args.keypoints, np.float32),
                    np.ones(args.keypoints, np.float32),
                ],
                axis=-1,
            )
            colors = np.squeeze(cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR))
            colors = (colors * 255).astype(np.uint8)
            for color, guesses in zip(colors, kps):
                for kp in guesses:
                    cv2.circle(image, tuple(kp[::-1]), 3, tuple(map(int, color)), -1)
            cv2.imwrite(os.path.join(args.render, f"{i}.png"), image)

    avg_inference_time = sum(result["inference_time"] for result in results) / len(
        results
    )
    avg_pnp_time = sum(result["pnp_time"] for result in results) / len(results)
    print(
        f"Average inference time: {int(avg_inference_time* 1000)}ms, average PnP time: {int(avg_pnp_time * 1000)}ms"
    )
    pose_errs = np.array([result["pose_error"] for result in results])
    position_errs = np.array([result["position_error_norm"] for result in results])
    utils.pose.display_geodesic_stats(pose_errs, position_errs)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
