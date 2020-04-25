from comet_ml import Experiment
import click
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import user_confirms
from datetime import datetime
from contextlib import ExitStack
import json
import tensorflow as tf
import numpy as np
import os
import shutil
import cv2

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import PoseRegressionModel
from . import utils


@click.group(help='TensorFlow Direct Pose Regression.')
def tf_pose_regression():
    pass


@tf_pose_regression.command(help="Train a model.")
@pass_train
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--comet", type=str, help="Enable comet integration under an experiment by this name", default=None)
@click.pass_context
def train(ctx, train: TrainInput, config, comet):
    # set base directory for model artifacts
    artifact_dir = LocalCache(global_cache.path / 'tf-feature-points').path if train.artifact_path is None \
        else train.artifact_path

    if os.path.exists(artifact_dir):
        if user_confirms('Artifact storage location contains old data. Overwrite?'):
            shutil.rmtree(artifact_dir)
        else:
            return ctx.exit()
    os.makedirs(artifact_dir)

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # load dataset mean and stdev
    # mean = np.load(str(train.dataset.path / 'mean.npy'))
    # stdev = np.load(str(train.dataset.path / 'stdev.npy'))

    with open(config, "r") as f:
        hyperparameters = json.load(f)

    # fill metadata
    metadata = {
        'architecture': 'feature_points_regression',
        'date_started_at': datetime.utcnow().isoformat() + "Z",
        'dataset_used': train.dataset.name,
        'config': hyperparameters
    }
    with open(artifact_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    experiment = None
    if comet:
        experiment = Experiment(workspace='seeker-rd', project_name='direct-pose-regression')
        experiment.set_name(comet)
        experiment.log_parameters(hyperparameters)
        experiment.set_os_packages()
        experiment.set_pip_packages()
        experiment.set_code()
        experiment.log_asset_data(metadata, file_name='metadata.json')

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    trainer = PoseRegressionModel(data_dir, hyperparameters)
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
            if "events.out.tfevents" in filename or ".h5" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    # evaluate
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.test())
        experiment.log_metric('pose_loss', _test(train.dataset.path, model_path))

    return TrainOutput(metadata, artifact_dir, model_path, extra_files, train.artifact_path is not None)


@tf_pose_regression.command(help="Evaluate a model (Keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
def test(train, model_path):
    _test(train.dataset.path, model_path)


def _test(dataset_path, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=PoseRegressionModel.pose_loss, optimizer=tf.keras.optimizers.SGD())

    cropsize = model.input.shape[1]
    test_data = utils.dataset_from_directory(os.path.join(dataset_path, 'test'), cropsize)
    test_data = test_data.map(
        lambda image, metadata: (
            tf.ensure_shape(image, [cropsize, cropsize, 3]),
            tf.ensure_shape(metadata["pose"], [4])
        )
    )
    """for image, pose in test_data:
        print(pose)
        im = (image.numpy() * 127.5 + 127.5).astype(np.uint8)
        cv2.imshow('a', im)
        cv2.waitKey(0)
    return"""
    test_data = test_data.batch(32)
    return model.evaluate(test_data)


