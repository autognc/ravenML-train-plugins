import click
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import user_confirms
from datetime import datetime
import json
import tensorflow as tf
import numpy as np
import os
import shutil
import cv2

from .train import PoseRegressionModel
from . import utils


@click.group(help='TensorFlow Direct Pose Regression.')
def tf_pose_regression():
    pass


@tf_pose_regression.command(help="Train a model.")
@pass_train
# @click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.pass_context
def train(ctx, train: TrainInput):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train".
    # After training, create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # load dataset mean and stdev
    # mean = np.load(str(train.dataset.path / 'mean.npy'))
    # stdev = np.load(str(train.dataset.path / 'stdev.npy'))

    hyperparameters = train.plugin_config

    # fill metadata
    train.metadata[train.plugin_metadata_field] = {
        'architecture': 'feature_points_regression',
        'config': hyperparameters
    }
    
    with open(artifact_dir / 'metadata.json', 'w') as f:
        json.dump(train.metadata, f, indent=2)

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    trainer = PoseRegressionModel(data_dir, hyperparameters)
    model_path = trainer.train(artifact_dir)

    # get Tensorboard files
    # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    #   but ravenML trashes this structure and just uploads the individual files to S3.
    extra_files = []
    for dirpath, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    return TrainOutput(train.metadata, artifact_dir, model_path, extra_files)


@tf_pose_regression.command(help="Evaluate a model (Keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
@click.pass_context
def eval(ctx, train, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=PoseRegressionModel.pose_loss, optimizer=tf.keras.optimizers.SGD())

    cropsize = model.input.shape[1]
    test_data = utils.dataset_from_directory(train.dataset.path / "test", cropsize)
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
    model.evaluate(test_data)


