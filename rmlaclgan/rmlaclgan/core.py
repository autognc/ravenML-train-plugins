from ravenml.train.options import kfold_opt, pass_train
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

from train import aclgan_train

@click.group(help='Top level command group description')
@click.pass_context
@kfold_opt
def aclgan(ctx, kfold):
    pass
    
parser.add_argument('--config', type=str, default='configs/male2female.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='aclgan', help="aclgan")

class Opts:
    def __init__(self, config=None, output_path=None, resume=None, trainer=None):
        self.config = config
        self.output_path = output_path
        self.resume = resume
        self.trainer = trainer
        
@aclgan.command()
@pass_train
@click.option("--comet", type=str, help="Enable comet integration under an experiment by this name", default=None)
@click.option('--config', type=str, default='configs/male2female.yaml', help='Path to the config file.')
@click.option('--output_path', type=str, default='./', help="outputs path")
@click.option("--resume", action="store_true")
@click.option('--trainer', type=str, default='aclgan', help="aclgan")
@click.pass_context
def train(ctx, train: TrainInput, comet, config, output_path, resume, trainer):
    # If the context (ctx) has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train"
    # after training, create an instance of TrainOutput and return it

    opts = Opts(config, output_path, resume, trainer)

    aclgan_train(opts)



    # artifact_dir = train.artifact_path

    # # set dataset directory
    # data_dir = train.dataset.path / "splits" / "complete" / "train"
    # keypoints_path = train.dataset.path / "keypoints.npy"

    # hyperparameters = train.plugin_config

    # keypoints_3d = np.load(keypoints_path)

    # # fill metadata
    # train.plugin_metadata["architecture"] = "keypoints_regression"
    # train.plugin_metadata["config"] = hyperparameters

    # experiment = None
    # if comet:
    #     experiment = Experiment(
    #         workspace="seeker-rd", project_name="keypoints-pose-regression"
    #     )
    #     experiment.set_name(comet)
    #     experiment.log_parameters(hyperparameters)
    #     experiment.set_os_packages()
    #     experiment.set_pip_packages()

    # # run training
    # print("Beginning training. Hyperparameters:")
    # print(json.dumps(hyperparameters, indent=2))
    # trainer = KeypointsModel(data_dir, hyperparameters, keypoints_3d)
    # with ExitStack() as stack:
    #     if experiment:
    #         stack.enter_context(experiment.train())
    #     model_path = trainer.train(artifact_dir, experiment)
    # if experiment:
    #     experiment.end()

    # # get Tensorboard files
    # # FIXME: The directory structure is very important for interpreting the Tensorboard logs
    # #   (e.x. phase_0/train/events.out.tfevents..., phase_1/validation/events.out.tfevents...)
    # #   but ravenML trashes this structure and just uploads the individual files to S3.
    # extra_files = []
    # for dirpath, _, filenames in os.walk(artifact_dir):
    #     for filename in filenames:
    #         if "events.out.tfevents" in filename:
    #             extra_files.append(os.path.join(dirpath, filename))

    # return TrainOutput(model_path, extra_files)
    return TrainOutput()