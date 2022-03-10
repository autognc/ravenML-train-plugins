from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from comet_ml import Experiment
from contextlib import ExitStack
import numpy as np
import click
import json
import os
from .train import HRNET

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
    data_dir = train.dataset.path / "splits" / "complete" / "train"
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
