import click
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from datetime import datetime
import json
import glob
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import FeaturePointsModel


@click.group(help='TensorFlow Feature Point Regression.')
def tf_feature_points():
    pass


HYP_TO_USE = [
    {
        'learning_rate': 4e-3,
        'learning_rate_2': 1e-5,
        'lr_decay_factor': 10,
        'dropout': 0.6,
        'batch_size': 128,
        'optimizer': 'RMSProp',
        'momentum': 0,
        'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 100,
        'regression_head_size': 1024,
        'classification_head_size': 256
     },
    {
        'learning_rate': 2e-3,
        'lr_decay_factor': 20,
        'dropout': 0.6,
        'batch_size': 128,
        'optimizer': 'RMSProp',
        'momentum': 0,
        'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 50,
        'regression_head_size': 1024,
        'classification_head_size': 256
    },
    {
        'learning_rate': 2e-3,
        'lr_decay_factor': 10,
        'dropout': 0.6,
        'batch_size': 128,
        'optimizer': 'SGD',
        'momentum': 0.5,
        'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 50,
        'regression_head_size': 1024,
        'classification_head_size': 256,
    },
    {
        'learning_rate': 2e-3,
        'lr_decay_factor': 20,
        'dropout': 0.6,
        'batch_size': 128,
        'optimizer': 'SGD',
        'momentum': 0.5,
        'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 50,
        'regression_head_size': 1024,
        'classification_head_size': 256,
    }
]


@tf_feature_points.command(help="Train a model.")
@pass_train
@click.pass_context
def train(ctx, train: TrainInput):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train".
    # After training, create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = LocalCache(global_cache.path / 'tf-feature-points').path if train.artifact_path is None \
        else train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # load number of feature points
    with open(train.dataset.path / 'feature_points.json', 'r') as f:
        feature_points = json.load(f)

    # load dataset mean and stdev
    mean = np.load(str(train.dataset.path / 'mean.npy'))
    stdev = np.load(str(train.dataset.path / 'stdev.npy'))

    for i, hp in enumerate(HYP_TO_USE):
        # fill metadata
        metadata = {
            'architecture': 'feature_points_regression',
            'date_started_at': datetime.utcnow().isoformat() + "Z",
            #'dataset_used': train.dataset.metadata,
            'feature_points': feature_points
        }
        trainer = FeaturePointsModel(data_dir, feature_points, mean, stdev, hp)
        metadata.update(trainer.hp)

        with open(artifact_dir / f'model_{i}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # run training
        logdir = str(artifact_dir / f'logs_{i}')
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        model = trainer.train(logdir=logdir)
        model_path = artifact_dir / f'model_{i}.h5'
        model.save(str(model_path.absolute()))

    # return TrainOutput
    return None
    # return TrainOutput(metadata, artifact_dir, model_path, [], train.artifact_path is not None)


@tf_feature_points.command(help="Visualize a model.")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
@click.pass_context
def eval(ctx, train, model_path):
    if train.artifact_path is None:
        ctx.fail("Must provide an artifact path for the visualizations to be saved to.")

    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"average_distance_error": FeaturePointsModel.average_distance_error})
    model.summary()

    with open(train.dataset.path / 'feature_points.json', 'r') as f:
        feature_points = json.load(f)

    image_filenames = glob.glob(str(train.dataset.path / "test" / "image_*"))
    meta_filenames = glob.glob(str(train.dataset.path / "test" / "meta_*"))
    images = []
    truths = []
    for image_filename, meta_filename in zip(sorted(image_filenames), sorted(meta_filenames)):
        with open(meta_filename, "r") as f:
            metadata = json.load(f)
        centroid = metadata["truth_centroids"]["barrel_center"]
        imsize = model.input_shape[1]
        half_imsize = imsize // 2
        image = cv2.imread(image_filename)
        image = image[centroid[0] - half_imsize:centroid[0] + half_imsize, centroid[1] - half_imsize:centroid[1] + half_imsize]
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        images.append(image)

        truth = [metadata["truth_centroids"][f][0] - centroid[0] + half_imsize for f in feature_points]\
            + [metadata["truth_centroids"][f][1] - centroid[1] + half_imsize for f in feature_points]
        truths.append(truth)

    images = np.array(images)
    truths = np.array(truths)
    predictions = model.predict(images)
    for image, prediction in zip(images, predictions):
        image = (image * 127.5 + 127.5).astype(np.uint8)
        predicted_points = prediction.reshape(2, len(feature_points)).transpose()
        for point_name, point in zip(feature_points, predicted_points):
            coord = tuple(point[::-1])
            cv2.circle(image, coord, 5, (0, 0, 255), -1)
            cv2.putText(image, point_name, coord, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.imshow('a', image)
        cv2.waitKey(0)

