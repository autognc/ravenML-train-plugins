"""
Evaluate on a directory of test data.
"""

import tensorflow as tf
import numpy as np
import os
import click
import json
import cv2
import tqdm
from .. import utils


def make_serializable(n):
    if not isinstance(n, np.ndarray):
        n = n.numpy()
    if isinstance(n, bytes):
        return n.decode("utf-8")
    return n.tolist()


@click.command(help="Evaluate a model (Keras .h5 format).")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.argument("object_name", type=str)
@click.option(
    "-k",
    "--keypoints",
    help='Path to 3D reference points .npy file (optional). If omitted, looks for "{directory}/keypoints.npy"',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-f",
    "--focal_length",
    type=float,
    help="Focal length (in pixels). Overrides focal length from truth data. Required if the truth data does not have focal length information.",
)
@click.option(
    "--flip",
    is_flag=True,
    help="Use more lenient error metric with barrel flip minimization",
)
@click.option("-n", "--num", type=int, help="Number of images to process (optional)")
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=False),
    help="Directory to write output (optional)",
)
@click.option(
    "-r",
    "--render",
    type=click.Path(file_okay=False),
    help="Directory to store keypoint renders (optional)",
)
def main(model_path, directory, object_name, keypoints, focal_length, flip, num, output, render):
    if render:
        os.makedirs(render, exist_ok=True)

    if output:
        os.makedirs(output, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)
    nb_keypoints = model.output.shape[-1] // 2
    crop_size = model.input.shape[1]

    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        directory, crop_size, nb_keypoints=nb_keypoints, focal_length=focal_length, object_name=object_name,
    )
    if num:
        data = data.take(num)
    data = data.batch(32)

    results = []
    errs_rot = []
    errs_pos = []
    errs_by_keypoint = []
    fls = []
    stdevs = []
    num_inliers = []
    bbox_sizes = []
    for image_batch, truth_batch in tqdm.tqdm(data):
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch_cropped = kps_batch * (crop_size // 2) + (crop_size // 2)
        kps_batch_uncropped = (
            kps_batch
            / 2
            * truth_batch["bbox_size"][
                :,
                None,
                None,
                None,
            ]
            + truth_batch["centroid"][:, None, None, :]
        )
        for i, (image, kps_cropped, kps_uncropped) in enumerate(
            zip(image_batch, kps_batch_cropped.numpy(), kps_batch_uncropped.numpy())
        ):
            fls.append(truth_batch["focal_length"][i])
            bbox_sizes.append(truth_batch["bbox_size"][i])

            r_vec, t_vec, inliers = utils.pose.solve_pose(
                ref_points,
                kps_uncropped,
                [truth_batch["focal_length"][i], truth_batch["focal_length"][i]],
                truth_batch["imdims"][i],
                ransac=True,
                reduce_mean=False,
                return_inliers=True,
            )
            num_inliers.append(inliers)
            errs_rot.append(
                utils.pose.geodesic_error(r_vec, truth_batch["pose"][i], flip=flip)
            )
            errs_pos.append(
                utils.pose.position_error(t_vec, truth_batch["position"][i])
            )

            kps_by_kp = kps_cropped.reshape(-1, nb_keypoints, 2).transpose([1, 0, 2])
            stdevs.append(
                np.linalg.norm(
                    kps_by_kp - kps_by_kp.mean(axis=1, keepdims=True), axis=2
                ).mean()
            )

            if render:
                image = tf.cast(image * 127.5 + 127.5, tf.uint8).numpy()
                utils.vis.vis_keypoints(
                    image, kps_cropped, err_rot=errs_rot[-1], err_pos=errs_pos[-1][1]
                )
                cv2.imwrite(
                    os.path.join(
                        render,
                        f"{truth_batch['image_id'][i].numpy().decode('utf-8')}.png",
                    ),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )

            result = {
                "rotation": truth_batch["pose"][i],
                "translation": truth_batch["position"][i],
                # "keypoints": truth_batch["keypoints"][i],
                "detected_rotation": utils.pose.to_rotation(r_vec).as_quat()[
                    [3, 0, 1, 2]
                ],
                "detected_translation": np.squeeze(t_vec),
                "detected_keypoints": kps_uncropped,
            }
            result = utils.data.recursive_map_dict(result, make_serializable)
            results.append(result)

    utils.pose.display_keypoint_stats(errs_by_keypoint)
    utils.pose.display_geodesic_stats(np.array(errs_rot), np.array(errs_pos).T)

    if output:
        with open(f"{output}/results.json", "w") as f:
            json.dump(results, f)
        np.save(f"{output}/errs_rot.npy", np.array(errs_rot))
        np.save(f"{output}/errs_pos.npy", np.array(errs_pos).T)
        np.save(f"{output}/errs_keypoint.npy", np.array(errs_by_keypoint))
        np.save(f"{output}/stdevs.npy", np.array(stdevs))
        np.save(f"{output}/fls.npy", np.array(fls))
        np.save(f"{output}/inliers.npy", np.array(num_inliers))
        np.save(f"{output}/bbox_sizes.npy", np.array(bbox_sizes))
