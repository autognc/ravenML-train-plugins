import traceback
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
import sys
import time
import shutil

from ravenml.utils.local_cache import LocalCache, global_cache
from .train import FeaturePointsModel


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
    def decode_json(filename):
        filename = filename.numpy()
        with open(filename, "r") as f:
            metadata = json.load(f)
        return [
            tf.convert_to_tensor(metadata["truth_centroids"]["barrel_center"], dtype=tf.float32),
            tf.convert_to_tensor(metadata["truth_centroids"]["barrel_top"], dtype=tf.float32),
            tf.convert_to_tensor(metadata["truth_centroids"]["barrel_bottom"], dtype=tf.float32),
            tf.convert_to_tensor(metadata["truth_centroids"]["panel_left"], dtype=tf.float32),
            tf.convert_to_tensor(metadata["truth_centroids"]["panel_right"], dtype=tf.float32),
            tf.convert_to_tensor(metadata["pose"], dtype=tf.float32)
        ]

    def parse_fn(image_filename, meta_filename):
        image_data = tf.io.read_file(image_filename)
        [centroid, top, bottom, left, right, pose] = tf.py_function(decode_json, [meta_filename], [tf.float32] * 6)
        barrel_length = tf.norm(top - bottom)
        panel_sep = tf.norm(left - right)
        bbox_size = tf.maximum(barrel_length, panel_sep)
        image = FeaturePointsModel.preprocess_image(image_data, centroid, bbox_size, cropsize)
        pose = tf.ensure_shape(pose, [4])
        return image, pose

    image_files = tf.data.Dataset.list_files(os.path.join(dir_path, "image_*"), shuffle=False)
    meta_files = tf.data.Dataset.list_files(os.path.join(dir_path, "meta_*"), shuffle=False)
    return tf.data.Dataset.zip((image_files, meta_files)).map(parse_fn, num_parallel_calls=4)


@tf_feature_points.command(help="Evaluate a model (keras .h5 format).")
@click.argument('model_path', type=click.Path(exists=True))
@pass_train
@click.pass_context
def eval(ctx, train, model_path):
    if train.artifact_path is None:
        ctx.fail("Must provide an artifact path for the visualizations to be saved to.")

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=FeaturePointsModel.pose_loss)


    """for image, pose in test_data:
        print(image)
        print(pose)
        im = (image.numpy() * 127.5 + 127.5).astype(np.uint8)
        cv2.imshow('a', im)
        cv2.waitKey(0)"""
    test_data = get_directory_dataset(train.dataset.path / "test", model.input.shape[1]).batch(32)
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
    tf.compat.v1.disable_eager_execution()
    model = tf.compat.v2.saved_model.load(model_path)
    cropsize = model.__call__.concrete_functions[0].inputs[0].shape[1]
    data = get_directory_dataset(directory, cropsize).take(num)
    iterator = tf.compat.v1.data.make_one_shot_iterator(data)

    image_op, truth_pose_op = iterator.get_next()
    image_op = tf.cast(image_op * 127.5 + 127.5, tf.uint8)

    image_placeholder = tf.compat.v1.placeholder(shape=[cropsize, cropsize, 3], dtype=tf.uint8)
    pose_op = model(image_placeholder)

    detected_pose_placeholder = tf.compat.v1.placeholder(shape=[4], dtype=tf.float32)
    pose_loss_op = FeaturePointsModel.pose_loss(truth_pose_op, detected_pose_placeholder)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        try:
            i = 0
            results = []
            while True:
                print("Image ", i)
                image = sess.run(image_op)
                start_time = time.time()
                detected_pose = sess.run(pose_op, feed_dict={image_placeholder: image})
                time_elapsed = time.time() - start_time
                print("Time: ", int(time_elapsed * 1000), "ms")
                truth_pose, pose_loss = sess.run([truth_pose_op, pose_loss_op], feed_dict={detected_pose_placeholder: detected_pose})

                result = {
                    'truth_pose': truth_pose.tolist(),
                    'detected_pose': detected_pose.tolist(),
                    'pose_error': float(pose_loss),
                    'time': time_elapsed,
                    'num': i
                }
                results.append(result)
                i += 1
        except tf.errors.OutOfRangeError:
            pass

    avg_time = sum(result['time'] for result in results) / len(results)
    avg_error = sum(result['pose_error'] for result in results) / len(results)
    print(f"Average time: {int(avg_time * 1000)}ms, average error: {avg_error}")
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
