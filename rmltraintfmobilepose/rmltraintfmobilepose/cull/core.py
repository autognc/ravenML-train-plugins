from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from comet_ml import Experiment
from contextlib import ExitStack
import numpy as np
import click
import json
import os
from pathlib import Path
from .train import train_cullnet
from .. import scripts
import pkgutil
import importlib
from .helpers import prepare_for_training

@click.group(help="TensorFlow Keypoints Regression.")
def tf_cullnet():
    pass


for _, name, _ in pkgutil.iter_modules(scripts.__path__):
    tf_cullnet.add_command(
        importlib.import_module(f"{scripts.__name__}.{name}").main, name=name
    )


@tf_cullnet.command(help="Train a model.")
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
    data_dir = train.dataset.path 
    

    hps = train.plugin_config
    keypoints_path = hps.get("keypoints_path",  train.dataset.path / "keypoints.npy")
    if not os.path.exists(keypoints_path):
        raise ValueError("No valid path to keypoints.npy file supplied")
    hps = prepare_for_training(hps)

    cache = Path(hps.get("cache", artifact_dir))
    os.makedirs(cache, exist_ok=True)
    # fill metadata
    train.plugin_metadata["architecture"] = "cullnet"
    train.plugin_metadata["config"] = hps

    experiment = None
    comet = hps.get("comet")

    if comet:
        experiment = Experiment(
            workspace="seeker-rd", 
            project_name="cullnet-error-prediction",
            auto_metric_logging=True,
            auto_param_logging=True,
            log_graph=True,
        )
        experiment.set_name(comet)
        experiment.log_parameters(hps)
        experiment.set_os_packages()
        experiment.set_pip_packages()
    # run training
    print("Beginning training. Hyperparameters:")
    print(json.dumps(hps, indent=2))
    with ExitStack() as stack:
        if experiment:
            stack.enter_context(experiment.train())
        model_path, extra_files = train_cullnet(
            hps["model_path"],
            data_dir,
            artifact_dir,
            hps["object_name"],
            hps["stl_path"],
            keypoints_path,
            hps["pnp_focal_length"],
            hps["error_metric"],
            hps["mask_mode"],
            hps["model_type"],
            hps.get("eval_trained_cull_model"),
            cache,
            experiment,
            hps,
        )

    return TrainOutput(model_path, extra_files)