from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import user_confirms
from datetime import datetime
import matplotlib.pyplot as plt
from comet_ml import Experiment
from contextlib import ExitStack
import tensorflow as tf
import numpy as np
import random
import shutil
import click
import json
import cv2
import os

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import KeypointsModel
from . import utils


@click.group(help='TensorFlow Keypoints Regression.')
def tf_keypoints():
    pass


@tf_keypoints.command(help="Train a model.")
@pass_train
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--comet", type=str, help="Enable comet integration under an experiment by this name", default=None)
@click.pass_context
def train(ctx, train: TrainInput, config, comet):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train".
    # After training, create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = (LocalCache(global_cache.path / 'tf-keypoints').path if train.artifact_path is None \
        else train.artifact_path) / 'artifacts'

    if os.path.exists(artifact_dir):
        if user_confirms('Artifact storage location contains old data. Overwrite?'):
            shutil.rmtree(artifact_dir)
        else:
            return ctx.exit()
    os.makedirs(artifact_dir)

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"
    keypoints_path = train.dataset.path / "keypoints.npy"

    with open(config, "r") as f:
        hyperparameters = json.load(f)

    keypoints_3d = np.load(keypoints_path)

    # fill metadata
    metadata = {
        'architecture': 'keypoints_regression',
        'date_started_at': datetime.utcnow().isoformat() + "Z",
        'dataset_used': train.dataset.name,
        'config': hyperparameters
    }
    with open(artifact_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    experiment = None
    if comet:
        experiment = Experiment(workspace='seeker-rd', project_name='keypoints-pose-regression')
        experiment.set_name(comet)
        experiment.log_parameters(hyperparameters)
        experiment.set_os_packages()
        experiment.set_pip_packages()
        experiment.log_asset_data(metadata, name='metadata.json')

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    trainer = KeypointsModel(data_dir, hyperparameters, keypoints_3d)
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.train())
        model_path = trainer.train(artifact_dir, experiment)

    # get Tensorboard files
    # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    #   but ravenML trashes this structure and just uploads the individual files to S3.
    extra_files = []
    for dirpath, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    return TrainOutput(metadata, artifact_dir, model_path, extra_files, train.artifact_path is not None)


@tf_keypoints.command(help="Evaluate a model (Keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--pnp_focal_length', default=1422)
@click.option('--plot', is_flag=True)
@click.option('--render_poses', is_flag=True)
@pass_train
@click.pass_context
def eval(ctx, train, model_path, pnp_focal_length=1422, plot=False, render_poses=False):
    errs = []
    errs_by_keypoint = []
    img_cnt = 0
    for ref_points, pose_batch, images, kps_pred_batch, kps_truth_batch \
            in utils.yield_eval_batches(train.dataset.path, model_path):
        n = images.shape[0]
        for i in range(n):
            image = images[i]
            kps = kps_pred_batch[i]
            kps_true = kps_truth_batch[i]
            # FIXME: don't hardcode image resolution
            r_vec, t_vec, cam_matrix, coefs = utils.calculate_pose_vectors(
                ref_points, kps, 
                pnp_focal_length, 1024)
            errs.append(utils.geodesic_error(r_vec, pose_batch[i]))
            errs_by_keypoint.append([np.linalg.norm(kps_true[i] - kps[i]) for i in range(len(kps))])
            if render_poses:
                for kp_idx in range(len(kps)):
                    y = int(kps[kp_idx, 0])
                    x = int(kps[kp_idx, 1])
                    ay = int(kps_true[kp_idx, 0])
                    ax = int(kps_true[kp_idx, 1])
                    cv2.circle(image, (x, y), 5, (255, 0, 255), -1)
                    cv2.circle(image, (ax, ay), 5, (255, 255, 255), -1)
                    cv2.line(image, (x, y), (ax, ay), (255, 0, 0), 3)
                landmarks = cv2.projectPoints(np.array([
                    [0, 5, -3.18566],
                    [0, -5, -3.18566],
                    [0, 0, -3.18566],
                    [0, 0, 3.18566],
                ], np.float32), r_vec, t_vec, cam_matrix, coefs)[0].squeeze()
                p1, p2, p3, p4 = landmarks
                cv2.line(image, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 255), 6)
                cv2.line(image, (p3[1], p3[0]), (p4[1], p4[0]), (255, 0, 255), 6)
                cv2.line(image, (p1[1], p1[0]), (p4[1], p4[0]), (255, 0, 255), 6)
                cv2.line(image, (p2[1], p2[0]), (p4[1], p4[0]), (255, 0, 255), 6)
                cv2.imwrite('pose-render-{:04d}.png'.format(img_cnt), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            img_cnt += 1

    _display_keypoint_stats(errs_by_keypoint)
    _display_geodesic_stats('Model Preds', errs, plot=plot)


@tf_keypoints.command(help="Evaluate ground truth PnP.")
@click.option('--keypoints', default=20)
@click.option('--cropsize', default=250)
@click.option('--pnp_focal_length', default=1422)
@click.option('--swap_random_percent', default=0, help="Randomly swap keypoints to test pnp.")
@pass_train
@click.pass_context
def evalpnp(ctx, train, keypoints=20, cropsize=250, pnp_focal_length=1422, swap_random_percent=0):
    errs = []
    rand_swap_amt = int(swap_random_percent / 100 * keypoints)
    if rand_swap_amt > 0:
        print('WARN: Randomly swapping {} keypoints.'.format(rand_swap_amt))

    for image, ref_points, kps, pose in utils.yield_meta_examples(train.dataset.path, cropsize):
        if rand_swap_amt > 0:
            swaps = random.sample(list(range(keypoints)), k=rand_swap_amt * 2)
            for a, b in zip(swaps[::2], swaps[1::2]):
                kps[a], kps[b] = kps[b], kps[a]

        # FIXME: don't hardcode image resolution
        r_vec, t_vec, cam_matrix, coefs = utils.calculate_pose_vectors(
            ref_points[:keypoints], kps[:keypoints],
            pnp_focal_length, 1024)
        err = utils.geodesic_error(r_vec, pose)
        errs.append(err)

    _display_geodesic_stats('PnP on truth, |kps|={})'.format(keypoints), errs)


def _display_geodesic_stats(title, errs, plot=False):
    np.save('pose_errs.npy', np.array(errs))
    print('\n---- Geodesic Error Stats ({}) ----'.format(title))
    stats = {
        'mean': np.mean(errs),
        'median': np.median(errs),
        'max': np.max(errs)
    }
    for label, val in stats.items():
        print('{:8s} = {:.3f} ({:.3f} deg)'.format(label, val, np.degrees(val)))
    if plot:
        plt.hist([np.degrees(val) for val in errs])
        plt.title(title)
        plt.show()


def _display_keypoint_stats(errs):
    np.save('keypoint_errs.npy', np.array(errs))
