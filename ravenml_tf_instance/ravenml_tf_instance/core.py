"""
Author(s):      Nihal Dhamani (nihaldhamani@gmail.com), 
                Carson Schubert (carson.schubert14@gmail.com)
Date Created:   04/10/2019

Core command group and commands for TF Instance Segmentation plugin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import warnings
warnings.filterwarnings("ignore")

import os
import click
import io
import sys
import yaml
import importlib
import re
from contextlib import ExitStack
from pathlib import Path
from datetime import datetime
from ravenml.options import verbose_opt
from ravenml.train.options import kfold_opt, pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.data.interfaces import Dataset
from ravenml.utils.question import cli_spinner, Spinner, user_selects, user_input
from ravenml.utils.plugins import fill_basic_metadata
from ravenml_tf_instance.utils.helpers import prepare_for_training, download_model_arch, instance_cache
import ravenml_tf_instance.validation.utils as utils
import ravenml_tf_instance.validation.stats as stats
from google.protobuf import text_format

# regex to ignore 0 indexed checkpoints
checkpoint_regex = re.compile(r'model.ckpt-[1-9][0-9]*.[a-zA-Z0-9_-]+')


### OPTIONS ###
# put any custom Click options you create here
no_comet_opt = click.option(
    '-c', '--no-comet', is_flag=True,
    help='Disable comet on this training run.'
)

no_validate_opt = click.option(
    '--no-validate', is_flag=True,
    help='Do not automatically run validation after training.'
)

### COMMANDS ###
@click.group(help='TensorFlow Object Detection with instance segmentation.')
@click.pass_context
def tf_instance(ctx):
    pass
    
@tf_instance.command(help='Train a model.')
@no_validate_opt
@no_comet_opt
@verbose_opt
# @kfold_opt
@pass_train
@click.pass_context
def train(ctx, train: TrainInput, verbose: bool, no_comet: bool, no_validate: bool):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train"
    
    # NOTE: after training, you must create an instance of TrainOutput and return it
    # import necessary libraries
    cli_spinner("Importing TensorFlow...", _import_od)
    if verbose:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
    
    # create training metadata dict and populate with basic information
    metadata = {}
    fill_basic_metadata(metadata, train.dataset)

    # set base directory for model artifacts 
    base_dir = instance_cache.path / 'temp' if train.artifact_path is None \
                    else train.artifact_path
 
    # load model choices from YAML
    models = {}
    models_path = os.path.dirname(__file__) / Path('utils') / Path('model_info.yml')
    with open(models_path, 'r') as stream:
        try:
            models = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # prompt for model selection
    model_name = user_selects('Choose model', models.keys())
    # grab fields and add to metadata
    model = models[model_name]
    model_type = model['type']
    model_url = model['url']
    metadata['architecture'] = model_name
    
    # download model arch
    arch_path = download_model_arch(model_url)

    # prepare directory for training/prompt for hyperparams
    if not prepare_for_training(base_dir, train.dataset.path, arch_path, model_type, metadata):
        ctx.exit('Training cancelled.')
        
    experiment = None
    if not no_comet:
        experiment = Experiment(workspace='seeker-rd', project_name='instance-segmentation')
        name = user_input('What would you like to name the comet experiment?:')
        experiment.set_name(name)
        experiment.log_parameters(metadata['hyperparameters'])
        experiment.set_git_metadata()
        experiment.set_os_packages()
        experiment.set_pip_packages()
    
    # get number of training steps
    num_train_steps = metadata['hyperparameters']['train_steps']
    try:
        num_train_steps = int(num_train_steps)
    except Exception as e:
        raise e

    model_dir = os.path.join(base_dir, 'models/model')
    pipeline_config_path = os.path.join(base_dir, 'models/model/pipeline.config')

    config = tf.estimator.RunConfig(model_dir=model_dir)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(None),
        pipeline_config_path=pipeline_config_path,
        train_steps=num_train_steps)
    
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        final_exporter_name='exported_model',
        eval_on_train_data=False)

    with ExitStack() as stack:
        if not no_comet:
            stack.enter_context(experiment.train())
        # actually train
        progress = Spinner('Training model...', 'magenta')
        if not verbose:
            progress.start()
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
        if not verbose:
            progress.succeed('Training model...Complete.')
        
    # final metadata and return of TrainOutput object
    metadata['date_completed_at'] = datetime.utcnow().isoformat() + "Z"
    
    # get extra config files
    extra_files, frozen_graph_path = _get_paths_for_extra_files(base_dir)
    model_path = frozen_graph_path
    local_mode = train.artifact_path is not None
    
    if not no_validate:
        with ExitStack() as stack:
            if not no_comet:
                stack.enter_context(experiment.validate())
                
            label_path = extra_files[-1]
            dev_path = train.dataset.path / 'splits/standard/dev'
            output_path = base_dir / 'validation'

            save_visualizations = False

            category_index = utils.get_categories(str(label_path))
            print("loaded label map")

            image_paths, mask_paths, metadata_paths, color_paths = utils.get_image_paths(dev_path)
            print("loaded image paths")

            images = utils.load_images_from_paths(image_paths)
            print("loaded images into array")

            masks = utils.load_masks_from_paths(mask_paths)
            print("loaded masks into array")

            colors = utils.load_colors_from_path(color_paths, category_index)
            print("loaded color labels into array")

            all_truths = utils.get_truth_masks(masks, colors, category_index)
            print("calculated truth values from masks")

            graph = utils.get_default_graph(str(model_path))
            print("loaded model graph")

            print("running inference for {} images..".format(str(len(images))))
            outputs, times = utils.run_inference_for_multiple_images(images, graph)
            print("inference done")

            all_detections = utils.convert_inference_output_to_detected_objects(category_index, outputs)
            print("converted inference outputs to detected objects")

            confidence, accuracy, recall, precision, iou, parameters = stats.calculate_statistics(all_truths, all_detections, category_index)
            print('calculated model performance')

            stats.write_stats_to_json(confidence, accuracy, recall, precision, iou, parameters, times, category_index, output_path)
            print('wrote model performance to json file')

            if save_visualizations:
                utils.visualize_and_save(images, all_detections, output_path)
                print("saved all visualizations")
                
            extra_files.append(output_path / 'stats.json')
            
            experiment.log_asset(output_path / 'stats.json')
                    
    result = TrainOutput(metadata, base_dir, model_path, extra_files, local_mode)
    return result
    

### HELPERS ###
def _get_paths_for_extra_files(artifact_path: Path):
    """Returns the filepaths for all checkpoint, config, and pbtxt (label)
    files in the artifact directory. Gets filepath for the exported inference
    graph.

    Args:
        artifact_path (Path): path to training artifacts
    
    Returns:
        list: list of Paths that point to files
    """
    extras = []
    # get checkpoints
    extras_path = artifact_path / 'models' / 'model'
    files = os.listdir(extras_path)

    # path to saved_model.pb
    saved_model_path = extras_path / 'export' / 'exported_model'
    saved_model_path = saved_model_path / os.listdir(saved_model_path)[0] / 'saved_model.pb'

    # path to label map
    labels_path = artifact_path / 'data' / 'label_map.pbtxt'

    checkpoints = [f for f in files if checkpoint_regex.match(f)]

    # calculate the max checkpoint
    max_checkpoint = 0
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint.split('-')[1].split('.')[0])
        if checkpoint_num > max_checkpoint:
            max_checkpoint = checkpoint_num

    ckpt_prefix = 'model.ckpt-' + str(max_checkpoint)
    checkpoint_path = extras_path / ckpt_prefix
    pipeline_path = extras_path / 'pipeline.config'
    exported_dir = artifact_path / 'frozen_model'

    # export frozen inference graph
    _export_frozen_inference_graph(str(pipeline_path), str(checkpoint_path), str(exported_dir))

    # append files to include in extras directory
    extras = [extras_path / f for f in checkpoints]
    extras.append(pipeline_path)
    extras.append(extras_path / 'graph.pbtxt')
    extras.append(saved_model_path)

    # path to exported frozen inference model
    frozen_graph_path = exported_dir / 'frozen_inference_graph.pb'

    # append event checkpoints for tensorboard
    for f in os.listdir(extras_path):
        if f.startswith('events.out'):
            extras.append(extras_path / f)

    for f in os.listdir(extras_path / 'eval_0'):
        if f.startswith('events.out'):
            extras.append(extras_path / 'eval_0' / f)

    extras.append(labels_path)
    return extras, frozen_graph_path


def _export_frozen_inference_graph(pipeline_config_path, checkpoint_path, output_directory):
    """Exports frozen inference graph from model checkpoints

    Args: 
        pipeline_config_path (str): path to pipeline config file
        checkpoint_path (str): path to checkpoint prefix with highest steps
            e.g. the checkpoint_path for /model/model.ckpt-100.index is 
            /model/model.ckpt-100
        output_directory (str): directory where the frozen_inference_graph will
            be outputted to
    """
    
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    
    input_shape = None
    exporter.export_inference_graph(
        'image_tensor', pipeline_config, checkpoint_path,
        output_directory, input_shape=input_shape,
        write_inference_graph=False)

# stdout redirection found at https://codingdose.info/2018/03/22/supress-print-output-in-python/
def _import_od():
    """ Imports the necessary libraries for object detection training.
    Used to avoid importing them at the top of the file where they get imported
    on every ravenML command call, even those not to this plugin.
    
    Also suppresses warning outputs from the TF OD API.
    """
    # create a text trap and redirect stdout
    # to suppress printed warnings from object detection and tf
    text_trap = io.StringIO()
    sys.stdout = text_trap
    sys.stderr = text_trap
    
    # Calls to _dynamic_import below map to the following standard imports:
    #
    # import tensorflow as tf
    # from object_detection import model_hparams
    # from object_detection import model_lib
    # from object_detection import exporter
    # from object_detection.protos import pipeline_pb2
    _dynamic_import('tensorflow', 'tf')
    _dynamic_import('object_detection.model_hparams', 'model_hparams')
    _dynamic_import('object_detection.model_lib', 'model_lib')
    _dynamic_import('object_detection.exporter', 'exporter')
    _dynamic_import('object_detection.protos', 'pipeline_pb2', asfunction=True)
    
    # now restore stdout function
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
# this function is derived from https://stackoverflow.com/a/46878490
# NOTE: this function should be used in all plugins, but the function is NOT
# importable because of the use of globals(). You must copy the code.
def _dynamic_import(modulename, shortname = None, asfunction = False):
    """ Function to dynamically import python modules into the global scope.

    Args:
        modulename (str): name of the module to import (ex: os, ex: os.path)
        shortname (str, optional): desired shortname binding of the module (ex: import tensorflow as tf)
        asfunction (bool, optional): whether the shortname is a module function or not (ex: from time import time)
        
    Examples:
        Whole module import: i.e, replace "import tensorflow"
        >>> _dynamic_import('tensorflow')
        
        Named module import: i.e, replace "import tensorflow as tf"
        >>> _dynamic_import('tensorflow', 'tf')
        
        Submodule import: i.e, replace "from object_detction import model_lib"
        >>> _dynamic_import('object_detection.model_lib', 'model_lib')
        
        Function import: i.e, replace "from ravenml.utils.config import get_config"
        >>> _dynamic_import('ravenml.utils.config', 'get_config', asfunction=True)
        
    """
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = importlib.import_module(modulename)
    else:        
        globals()[shortname] = getattr(importlib.import_module(modulename), shortname)
