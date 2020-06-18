"""
Author(s):      Nihal Dhamani (nihaldhamani@gmail.com), 
                Carson Schubert (carson.schubert14@gmail.com)
Date Created:   12/06/2019

Core command group and commands for TF Bounding Box plugin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from comet_ml import Experiment
import os
import click 
import io
import sys
import yaml
import importlib
import re
import glob
import shutil
from contextlib import ExitStack
import traceback
from pathlib import Path
from datetime import datetime
from ravenml.options import verbose_opt
from ravenml.train.options import kfold_opt, pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml.utils.question import cli_spinner, user_selects, user_input
from ravenml.utils.plugins import fill_basic_metadata
from rmltraintunetfbbox.utils.helpers import prepare_for_training, download_model_arch, bbox_cache
import rmltraintunetfbbox.validation.utils as utils
from rmltraintunetfbbox.validation.model import BoundingBoxModel
from rmltraintunetfbbox.validation.stats import BoundingBoxEvaluator
from google.protobuf import text_format
import ray
from ray.tune import Trainable, grid_search, run_experiments
from ray.tune.schedulers import PopulationBasedTraining
from tensorflow.python.data.experimental import CheckpointInputPipelineHook
import datetime

# regex to ignore 0 indexed checkpoints
checkpoint_regex = re.compile(r'model.ckpt-[1-9][0-9]*.[a-zA-Z0-9_-]+')


### OPTIONS ###
# put any custom Click options you create here
comet_opt = click.option(
    '-c', '--comet', is_flag=True,
    help='Enable comet on this training run.'
)


### COMMANDS ###
@click.group(help='TensorFlow Object Detection with bounding boxes with tune optimization.')
@click.pass_context
def tf_bbox_tune(ctx):
    pass
    
@tf_bbox_tune.command(help='Train a model.')
@verbose_opt
@comet_opt
# @kfold_opt
@pass_train
@click.pass_context
def train(ctx, train: TrainInput, verbose: bool, comet: bool):
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
    base_dir = bbox_cache.path / 'temp' if train.artifact_path is None \
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
    print(base_dir)
    print(type(base_dir))
    ##### start of setup
    # prepare directory for training/prompt for hyperparams
    if not prepare_for_training(base_dir, train.dataset.path, arch_path, model_type, metadata):
        ctx.exit('Training cancelled.')

    model_dir = os.path.join(base_dir, 'models/model')
    models_dir = os.path.join(base_dir, 'models')
    pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
    train_spec = {
        "run": MyTrainableEstimator,
        "resources_per_trial": {
            "cpu": 5,
        },
        "stop": {
            "accuracy": 1.0,  # value of the loss to stop, check with attribute
            "training_iteration": 40,  # how many times train is invoked
        },
        "config": {
            "lr": grid_search([10 ** -2, 10 ** -5]),
            "base_dir": base_dir,
            "base_pipeline": pipeline_config_path,
            "arch_path": arch_path,
            "cur_dir" : Path(os.path.dirname(os.path.abspath(__file__)))
        },
        "num_samples": 10,
        'local_dir': models_dir,
        'checkpoint_at_end': True

    }

    ray.init()
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="accuracy",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": lambda: np.random.uniform(0, 1),
        })

    run_experiments({"pbt_estimator": train_spec}, scheduler=pbt)
"""
    ### start of train? can we split this up?
    # actually train
    with ExitStack() as stack:
        if comet:
            stack.enter_context(experiment.train())
        click.echo('Training model...')
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
        click.echo('Training complete')

    # final metadata and return of TrainOutput object
    metadata['date_completed_at'] = datetime.utcnow().isoformat() + "Z"

    # get extra config files
    extra_files, frozen_graph_path = _get_paths_for_extra_files(base_dir)
    model_path = str(frozen_graph_path)
    local_mode = train.artifact_path is not None
    if comet:
        experiment.log_asset(model_path)

    try:
        click.echo("Evaluating model...")
        with ExitStack() as stack:
            if comet:
                stack.enter_context(experiment.validate())

            # path to label_map.pbtxt
            label_path = str(extra_files[-1])
            test_path = str(train.dataset.path / 'test')
            output_path = str(base_dir / 'validation')

            image_dataset = utils.get_image_dataset(test_path)
            truth_data = list(utils.gen_truth_data(test_path))

            model = BoundingBoxModel(model_path, label_path)
            evaluator = BoundingBoxEvaluator(model.category_index)
            image_tensor = image_dataset.make_one_shot_iterator().get_next()
            with tf.Session() as sess:
                with model.start_session():
                    for i, (bbox, centroid) in enumerate(truth_data):
                        image = sess.run(image_tensor)
                        output, inference_time = model.run_inference_on_single_image(image)
                        evaluator.add_single_result(output, inference_time, bbox, centroid)

            evaluator.dump(os.path.join(output_path, 'validation_results.pickle'))
            if comet:
                experiment.log_asset('validation_results.pickle')
            evaluator.calculate_default_and_save(output_path)

            extra_files.append(os.path.join(output_path, 'stats.json'))
            extra_files.append(os.path.join(output_path, 'validation_results.pickle'))
            extra_files += glob.glob(os.path.join(output_path, '*_curve_*.png'))
            if comet:
                experiment.log_asset(os.path.join(output_path, 'stats.json'))
                for img in glob.glob(os.path.join(output_path, '*_curve_*.png')):
                    experiment.log_image(img)
    except Exception:
        metadata['validation_error'] = traceback.format_exc()

    if comet:
        experiment.log_asset_data(metadata, file_name="metadata.json")

    result = TrainOutput(metadata, base_dir, Path(model_path), extra_files, local_mode)
    return result"""

class MyTrainableEstimator(Trainable):
    """
    Example how to combine tf.Estimator with ray.tune population based training (PBT)
    example is based on:
    TensorFlow doc examples:
    https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py
    and estimator with raytune example by @sseveran:
    https://github.com/sseveran/ray-tensorflow-trainable/blob/master/estimator.py
    data loaded from script:
    https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
    """

    def _setup(self, config):
        """
        Setup your tensorflow model
        :param config:
        :return:
        """
        
        #### not sure why these libraries have to be reimported 
        cli_spinner("Importing TensorFlow...", _import_od)
        from pathlib import Path
        self.config = config
        base_dir  = self.config['base_dir']
        ###add comet config later
        self.base_pipeline_config_path = self.config['base_pipeline']
        
        # create models, model, eval, and train folders
        self.model_folder = base_dir / 'models' / datetime.datetime.now().strftime("%d_%b_%Y_%I_%M_%S_%f%p")
        eval_folder = self.model_folder / 'eval'
        train_folder = self.model_folder / 'train'
        os.makedirs(self.model_folder)
        os.makedirs(eval_folder)
        os.makedirs(train_folder)
        self.pipeline_config_path = os.path.join(self.model_folder, 'pipeline.config')
        
        
        base_config = pipeline_pb2.TrainEvalPipelineConfig()
       
        with tf.io.gfile.GFile(self.base_pipeline_config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, base_config) 

        base_config.train_config.fine_tune_checkpoint = str(self.model_folder / 'train'/ 'model.ckpt')
        base_config.train_config.optimizer.rms_prop_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate = self.config['lr']
        pipeline_contents = text_format.MessageToString(base_config)

        # output final configuation file for training
        with open(self.model_folder / 'pipeline.config', 'w') as file:
            file.write(pipeline_contents)
            # copy model checkpoints to our train folder
        checkpoint_folder = self.config['arch_path']
        checkpoint0_folder = self.config['cur_dir'] /'utils' / 'checkpoint_0'
        file1 = checkpoint_folder / 'model.ckpt.data-00000-of-00001'
        file2 = checkpoint_folder / 'model.ckpt.index'
        file3 = checkpoint_folder / 'model.ckpt.meta'
        file4 = checkpoint0_folder / 'model.ckpt-0.data-00000-of-00001'
        file5 = checkpoint0_folder / 'model.ckpt-0.index'
        file6 = checkpoint0_folder / 'model.ckpt-0.meta'
        shutil.copy2(file1, train_folder)
        shutil.copy2(file2, train_folder)
        shutil.copy2(file3, train_folder)
        shutil.copy2(file4, train_folder)
        shutil.copy2(file5, train_folder)
        shutil.copy2(file6, train_folder)
    
        # load starting checkpoint template and insert training directory path
        checkpoint_file = checkpoint0_folder / 'checkpoint'
        with open(checkpoint_file) as cf:
            checkpoint_contents = cf.read()
        checkpoint_contents = checkpoint_contents.replace('<replace>', str(train_folder))
        with open(train_folder / 'checkpoint', 'w') as new_cf:
            new_cf.write(checkpoint_contents)
        ####does this need to be different for each
        self.training_steps = 1000
        
        run_config = tf.estimator.RunConfig(
            model_dir=self.model_folder,
            save_summary_steps=100,
            save_checkpoints_secs=None,  # save checkpoint only before and after self.estimator.train()
            save_checkpoints_steps=self.training_steps,  # save both iterator checkpoint and model checkpoints after
            # same number of steps
            keep_checkpoint_max=None,  # avoid removing checkpoints
            keep_checkpoint_every_n_hours=None)
        
        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=run_config,
            sample_1_of_n_eval_examples=1,
            hparams=model_hparams.create_hparams(None),
            pipeline_config_path=self.pipeline_config_path,
            train_steps=self.training_steps)
    
        self.estimator = train_and_eval_dict['estimator']
        self.train_input_fn = train_and_eval_dict['train_input_fn']
        self.eval_input_fns = train_and_eval_dict['eval_input_fns']
        self.eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        self.predict_input_fn = train_and_eval_dict['predict_input_fn']
        self.train_steps = train_and_eval_dict['train_steps']

        self.train_spec, self.eval_specs = model_lib.create_train_and_eval_specs(
            self.train_input_fn,
            self.eval_input_fns,
            self.eval_on_train_input_fn,
            self.predict_input_fn,
            self.training_steps,
            final_exporter_name ='exported_model',
            eval_on_train_data=False)
        self.steps = 0

    def _train(self):
        # Run your training op for n iterations

        # possible to run evaluation in memory with:
        # evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
        # self.estimator, self.input_fn_eval)

        # CheckpointInputPipelineHook save data iterator to resume loading data
        # from checkpoint. Otherwise iterator is initialized every time new estimator is created and iteration starts
        # from start point which might cause overfitting for big data models. CheckpointInputPipelineHook(...) use
        # self.run_config.save_checkpoints_secs or save_checkpoints_steps to save iterator. for more control over
        # saving read more: https://www.tensorflow.org/api_docs/python/tf/contrib/data/CheckpointInputPipelineHook
        # self.datahook = CheckpointInputPipelineHook(self.estimator)
        # training
        tf.estimator.train_and_evaluate(self.estimator, self.train_spec, self.eval_specs[0])
        self.steps = self.steps + self.training_steps
        return metrics

    def _stop(self):
        self.estimator = None

    def _save(self, checkpoint_dir):
        """
         This function will be called if a population member is good enough to be exploited
        :param checkpoint_dir:
        :return:
        """
        lastest_checkpoint = self.estimator.latest_checkpoint()
        # lastest_checkpoint = tf.contrib.training.wait_for_new_checkpoint(
        #    checkpoint_dir=self.model_dir_full,
        #    last_checkpoint=self.estimator.latest_checkpoint(),
        #    seconds_to_sleep=0.01,
        #    timeout=60
        # )
        #
        tf.logging.info('Saving checkpoint {} for tune'.format(lastest_checkpoint))
        f = open(checkpoint_dir + '/path.txt', 'w')
        f.write(lastest_checkpoint)
        f.flush()
        f.close()
        return checkpoint_dir + '/path.txt'

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
    _dynamic_import('tensorflow', 'tf')
    _dynamic_import('object_detection.model_hparams', 'model_hparams')
    _dynamic_import('object_detection.model_lib', 'model_lib')
    _dynamic_import('object_detection.exporter', 'exporter')
    #_dynamic_import('object_detection.protos', 'pipeline_pb2')
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
