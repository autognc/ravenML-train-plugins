import traceback
import click
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from datetime import datetime
import json
import tensorflow as tf
import numpy as np
import os
import glob
import sys
import time
import shutil
import cv2

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import FeaturePointsModel


def recursive_map_dict(d, f):
    if isinstance(d, dict):
        return {k: recursive_map_dict(v, f) for k, v in d.items()}
    return f(d)


@click.group(help='TensorFlow Feature Point Regression.')
def tf_feature_points():
    pass


HYP_TO_USE = [
    {
        'learning_rate': 0.0045,
        'learning_rate_2': 0.0045,
        'pooling': 'avg',
        'fine_tune_start': 'block_16_expand',
        'fine_tune_start_2': 'block_15_expand',
        #'l2_reg': 0,
        'dropout': 0.5,
        'batch_size': 150,
        'optimizer': 'RMSProp',
        'optimizer_2': 'RMSProp',
        #'momentum': 0,
        #'nesterov_momentum': True,
        'crop_size': 224,
        'num_fine_tune_layers': 0,
        'epochs': 30,
        'epochs_2': 100,
        'regression_head_size': 1024
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

    try:
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
            #if os.path.exists(logdir):
                #shutil.rmtree(logdir)
            model = trainer.train(logdir=logdir)
            #model_path = artifact_dir / f'model_{i}.h5'
            #model.save(str(model_path.absolute()))
    except:
        traceback.print_exc()
    #sys.exit(0)
        
    # return TrainOutput
    return None
    # return TrainOutput(metadata, artifact_dir, model_path, [], train.artifact_path is not None)


def get_directory_dataset(dir_path, cropsize):
    def generator():
        image_files = sorted(glob.glob(os.path.join(dir_path, "image_*")))
        meta_files = sorted(glob.glob(os.path.join(dir_path, "meta_*")))
        for image_file, meta_file in zip(image_files, meta_files):
            image_data = tf.io.read_file(image_file)
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            metadata = recursive_map_dict(metadata, tf.convert_to_tensor)
            centroid = tf.cast(metadata["truth_centroids"]["barrel_center"], tf.float32)
            top = tf.cast(metadata["truth_centroids"]["barrel_top"], tf.float32)
            bottom = tf.cast(metadata["truth_centroids"]["barrel_bottom"], tf.float32)
            left = tf.cast(metadata["truth_centroids"]["panel_left"], tf.float32)
            right = tf.cast(metadata["truth_centroids"]["panel_right"], tf.float32)
            barrel_length = tf.norm(top - bottom)
            panel_sep = tf.norm(left - right)
            bbox_size = tf.maximum(barrel_length, panel_sep)
            image = FeaturePointsModel.preprocess_image(image_data, centroid, bbox_size, cropsize)
            yield image, metadata

    meta_file_0 = glob.glob(os.path.join(dir_path, "meta*"))[0]
    with open(meta_file_0, "r") as f:
        meta0 = json.load(f)
    dtypes = recursive_map_dict(meta0, lambda x: tf.convert_to_tensor(x).dtype)
    return tf.data.Dataset.from_generator(generator, (tf.float32, dtypes))


@tf_feature_points.command(help="Evaluate a model (keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
@click.pass_context
def eval(ctx, train, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=FeaturePointsModel.pose_loss)

    cropsize = model.input.shape[1]
    test_data = get_directory_dataset(train.dataset.path / "test", cropsize)
    test_data = test_data.map(
        lambda image, metadata: (
            tf.ensure_shape(image, [cropsize, cropsize, 3]),
            tf.ensure_shape(metadata["pose"], [4])
        )
    )
    test_data = test_data.batch(32)
    """for image, pose in test_data:
        print(pose)
        im = (image.numpy() * 127.5 + 127.5).astype(np.uint8)
        cv2.imshow('a', im)
        cv2.waitKey(0)"""
    model.evaluate(test_data)


@tf_feature_points.command(help="Freeze a model to Tensorflow format so that it can be served from anywhere.")
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def freeze(model_path, output_path):
    model = tf.keras.models.load_model(model_path, compile=False)

    # for some reason, directly trying to save the keras model isn't working,
    # so let's create a custom one as a workaround. This conveniently also
    # allows us to build in the preprocessing and get rid of the batch dimension
    class Module(tf.Module):
        def __init__(self, model):
            # directly copy over the variables to get rid of keras weirdness
            for layer in model.layers:
                for variable in layer.variables:
                    setattr(self, variable.name, variable)

        @tf.function(input_signature=[tf.TensorSpec(model.input.shape[1:], dtype=np.uint8)])
        def __call__(self, image):
            batch_im = tf.expand_dims(image, 0)
            normalized = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(batch_im, tf.float32))
            return tf.squeeze(model(normalized))

    module = Module(model)
    # call it once to create a trace
    #module(tf.convert_to_tensor(np.zeros(model.input.shape[1:], dtype=np.uint8)))
    # save
    tf.saved_model.save(module, output_path)


@tf_feature_points.command(help="Evaluate a model (TF format) on a directory of images.")
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('directory', type=click.Path(exists=True))
@click.option('--num', '-n', type=int, default=-1)
def process_directory(model_path, directory, num):
    model = tf.compat.v2.saved_model.load(model_path)
    cropsize = model.__call__.concrete_functions[0].inputs[0].shape[1]
    data = get_directory_dataset(directory, cropsize).take(num)

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
            'pose_error': make_serializable(FeaturePointsModel.pose_loss(metadata['pose'], detected_pose))
        })
        results.append(result)

    avg_time = sum(result['time'] for result in results) / len(results)
    avg_error = sum(result['pose_error'] for result in results) / len(results)
    print(f"Average time: {int(avg_time * 1000)}ms, average error: {avg_error}")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
