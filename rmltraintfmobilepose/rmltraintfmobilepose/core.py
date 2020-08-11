from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from comet_ml import Experiment
from contextlib import ExitStack
import numpy as np
import click
import json
import os
from .train import KeypointsModel
from . import scripts
import pkgutil
import importlib


@click.group(help="TensorFlow Keypoints Regression.")
def tf_mobilepose():
    pass


for _, name, _ in pkgutil.iter_modules(scripts.__path__):
    tf_mobilepose.add_command(
        importlib.import_module(f"{scripts.__name__}.{name}").main, name=name
    )


@tf_mobilepose.command(help="Train a model.")
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

    keypoints_3d = np.load(keypoints_path)

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
    trainer = KeypointsModel(data_dir, hyperparameters, keypoints_3d)
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.train())
        model_path = trainer.train(artifact_dir, experiment)
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
