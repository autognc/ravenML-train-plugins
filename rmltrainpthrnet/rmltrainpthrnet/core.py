import pathlib
from cv2 import split
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from comet_ml import Experiment
from contextlib import ExitStack
import numpy as np
import click
import json
import os
import tensorflow as tf
import cv2
from pathlib import Path
import shutil
import boto3

from ravenml.utils.local_cache import RMLCache
from .train import HRNET
from ravenml.utils.config import get_config

import pkgutil
import importlib
import yaml
from attrdict import AttrDict
import pytorch_lightning as pl

@click.group(help="Pytorch Keypoints Regression.")
def pt_hrnet():
    pass


@pt_hrnet.command(help="Train a model.")
@pass_train
@click.option(
    "--comet",
    type=str,
    help="Enable comet integration under an experiment by this name",
    default=None,
)
@click.pass_context
def train(ctx, train: TrainInput, comet):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # object creation, after which execution will fail as this means
    # the user did not pass a config. see ravenml core file train/commands.py for more detail

    # NOTE: after training, you must create an instance of TrainOutput and return it

    # set base directory for model artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    if os.path.exists(train.dataset.path / "splits"):
        print("Dataset is in TFRecord format, but PyTorch format is required. Use the convert-dataset command to switch.")
        return
    data_dir = train.dataset.path 
    keypoints_path = train.dataset.path / "keypoints.npy"

    hyperparameters = train.plugin_config
    defaults_path = os.path.join(os.path.dirname(__file__), "utils", "model_defaults.yml")
    with open(defaults_path, 'r') as stream:
        try:
            hyperparameters.update(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    keypoints_3d = np.load(keypoints_path)
    hyperparameters = AttrDict(hyperparameters) ## TODO: make sure all num_keypoints and Num_joints are the same
    # fill metadata
    train.plugin_metadata["architecture"] = "keypoints_regression"
    train.plugin_metadata["config"] = hyperparameters

    experiment = None
    if comet:
        experiment = Experiment(
            workspace="seeker-rd", project_name="keypoints-pose-regression"
        )
        experiment.set_name(comet)
        experiment.log_parameters(hyperparameters)
        experiment.set_os_packages()
        experiment.set_pip_packages()

    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hyperparameters, indent=2))
    model = HRNET(hyperparameters, data_dir, artifact_dir, keypoints_3d)
    trainer = pl.Trainer(gpus=1)
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.train())
        model_path = trainer.fit(model)
    if experiment:
        experiment.end()

    # get Tensorboard files
    # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    #   but ravenML trashes this structure and just uploads the individual files to S3.
    extra_files = []
    for dirpath, _, filenames in os.walk(artifact_dir):
        for filename in filenames:
            if "events.out.tfevents" in filename:
                extra_files.append(os.path.join(dirpath, filename))

    return TrainOutput(model_path, extra_files)

@pt_hrnet.command(help="Convert dataset from TensorFlow to PyTorch.")
@pass_train
@click.option(
    "--comet",
    type=str,
    help="Enable comet integration under an experiment by this name",
    default=None,
)
@click.pass_context
def convert_dataset(ctx, train: TrainInput, comet):
    dataset_name = train.config['dataset']
    dataset_path = RMLCache('datasets').path / Path(dataset_name)
    
    data_dir = dataset_path / "splits" / "complete" / "train"

    if dataset_name == "_pt":
        print("{} dataset does not need to be converted; it is already in PyTorch form, as indicated by '_pt' suffix.")
        return

    
    converted_dataset_name = dataset_name + "_pt"
    converted_dataset_path = RMLCache('datasets').path / converted_dataset_name

    ds_bucket = get_config()['dataset_bucket_name']
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(name=ds_bucket)



    if not os.path.isdir(converted_dataset_path):
        os.makedirs(converted_dataset_path)

    converted_dataset_files = bucket.objects.filter(Prefix=converted_dataset_name)
    if list(iter(converted_dataset_files)):
        print('Converted dataset available online, downloading dataset {}'.format(converted_dataset_name))
        for obj in converted_dataset_files:
            filename = os.path.basename(obj.key)
            full_filename = os.path.join(converted_dataset_path, filename)
            bucket.download_file(Key=obj.key, Filename=full_filename)
            if '.zip' in filename:
                extract_dir = filename[:filename.index('.zip')]
                shutil.unpack_archive(full_filename, converted_dataset_path / extract_dir)
                os.remove(full_filename)

        return 

    for item in os.listdir(dataset_path):
        if os.path.isfile(dataset_path / item):
            shutil.copy(dataset_path / item, converted_dataset_path)

    def convert_split(split_name: str):

        filenames = tf.io.gfile.glob(os.path.join(data_dir, f"{split_name}.record-*"))
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=16)

        features = {
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/object/keypoints": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/object/pose": tf.io.VarLenFeature(tf.float32),
            "image/imageset": tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/translation': tf.io.VarLenFeature(tf.float32)
        }

        dataset = dataset.map(
            lambda example: tf.io.parse_single_example(example, features),
            num_parallel_calls=16,
        )

        if not os.path.isdir(converted_dataset_path / split_name):
            os.makedirs(converted_dataset_path / split_name)

        def save_record(record):
            encoded_image = record.pop('image/encoded')
            image = tf.io.decode_image(encoded_image, channels=3).numpy()
            new_directory = converted_dataset_path / split_name
            image_name = record['image/filename'].numpy().decode('utf8')
            cv2.imwrite(str(new_directory / image_name), image)

            # data needs to be converted from EagerTensors to JSON serializable data formats
            for key, val in features.items():
                if key == 'image/encoded':
                    continue
                # get numpy array from either eager or sparse tensor
                if type(val) == tf.io.FixedLenFeature:
                    record[key] = record[key].numpy()
                elif type(val) == tf.io.VarLenFeature:
                    record[key] = record[key].values.numpy()
                
                # get datatype casting function
                t = None

                if val.dtype in [tf.int64, tf.int32, tf.int16, tf.int8]:
                    t = int
                elif val.dtype in [tf.float64, tf.float32, tf.float16]:
                    t = float
                elif val.dtype == tf.string:
                    t = lambda v: v.decode('utf8')
                # convert to list or singular of proper type
                if type(record[key]) == bytes or not hasattr(record[key], '__len__') or len(record[key]) == 1:
                    record[key] = t(record[key])
                else:
                    record[key] = [t(v) for v in record[key]]
            
            meta_name = image_name.replace('image', 'meta')
            extension_idx = meta_name.index('.')
            meta_name = meta_name[:extension_idx] + '.json'

            with open(str(new_directory / meta_name), 'w') as f:
                json.dump(record, f)

        for record in dataset:
            save_record(record)

    for split_name in ['train', 'test']:
        convert_split(split_name)
        shutil.make_archive(converted_dataset_path / split_name, 'zip', converted_dataset_path / split_name)

    for item in os.listdir(converted_dataset_path):
        if os.path.isfile(converted_dataset_path / item):
            bucket.upload_file(Filename=str(converted_dataset_path / item), Key=os.path.join(converted_dataset_name, item))
            if '.zip' in item:
                os.remove(converted_dataset_path / item)

