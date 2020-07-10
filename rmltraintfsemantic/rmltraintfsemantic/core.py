import click
import yaml
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from datetime import datetime
from pathlib import Path
import glob
import sys
import re
import os
import shutil

def parse_config(config):
    result = {}
    for arg in config:
        arg = list(arg.items())
        if len(arg) > 1:
            raise ValueError("Invalid config file, please only specify one key-value pair per list item")
        result[arg[0][0]] = str(arg[0][1])
    return result


def parse_deeplab_args(args):
    result = {}
    for arg in args:
        split = arg[2:].split("=")
        if len(split) != 2:
            raise ValueError(f"Improperly formatted deeplab arg {arg}, must be of form key=value")
        result[split[0]] = split[1]
    return result


def setup_dataset(dataset_path):
    from deeplab.datasets import data_generator
    # read number of classes from Jigsaw label map, this is bad but I'm not gonna pull out
    # a protobuf parser just to count the number of IDs
    with open(Path(dataset_path) / "label_map.pbtxt", "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)

    # set up data generator for our dataset
    dataset_info = data_generator.DatasetDescriptor(
        splits_to_sizes={
            'train': -1,  # these aren't actually used
            'test': -1,
        },
        num_classes=num_classes + 1,
        ignore_label=0,
    )
    data_generator._DATASETS_INFORMATION['custom'] = dataset_info

    return num_classes

@click.group(help='TensorFlow Semantic Segmentation.')
def tf_semantic():
    pass

@tf_semantic.command(help="Train a model.", context_settings=dict(ignore_unknown_options=True))
@pass_train
@click.argument('extra_deeplab_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def train(ctx, train: TrainInput, extra_deeplab_args):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which execution will fail as this means 
    # the user did not pass a config. see ravenml core file train/commands.py for more detail
    from deeplab import train as deeplab_train

    # set base directory for model artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # parse config file
    config_opts = parse_config(train.plugin_config)
    # parse extra deeplab args
    extra_opts = parse_deeplab_args(extra_deeplab_args)

    # union all options
    config_opts.update(extra_opts)

    # set up entry for "custom" dataset and load number of classes
    num_classes = setup_dataset(train.dataset.path)

    # fill metadata
    train.plugin_metadata['architecture'] = 'deeplab'
    train.plugin_metadata['num_classes'] = num_classes
    train.plugin_metadata['deplab_options'] = config_opts

    # fill sys.argv to be passed to deeplab
    sys.argv = [sys.argv[0]]
    sys.argv.append("--dataset=custom")
    sys.argv.append(f"--dataset_dir={data_dir.absolute()}")
    sys.argv.append(f"--train_logdir={artifact_dir.absolute()}")
    sys.argv.append("--train_split=train")
    sys.argv.append("--initialize_last_layer=False")
    sys.argv.append("--last_layers_contain_logits_only=True")
    sys.argv += [f"--{key}={value}" for key, value in config_opts.items()]

    # run deeplab
    try:
        with TFRecordFilenameConverter(data_dir.absolute()):
            deeplab_train.main(None)
    except KeyboardInterrupt:
        pass

    # return TrainOutput
    model_path = artifact_dir / "checkpoint"
    checkpoint_files = list(map(Path, (glob.glob(str(artifact_dir.absolute() / "*")))))
    return TrainOutput(Path(model_path), checkpoint_files)

# NOTE: eval and vis are NOT tested with the current setup. test if we start using deeplab again

@tf_semantic.command(help="Evaluate a model. Always use in local mode.", context_settings=dict(ignore_unknown_options=True))
@pass_train
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.argument('extra_deeplab_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def eval(ctx, train: TrainInput, checkpoint_dir, extra_deeplab_args):
    from deeplab import eval as deeplab_eval
    from deeplab import common as deeplab_common

    # stupid hack to prevent deeplab from skipping label reading when the split is called 'test'
    deeplab_common.TEST_SET = None

    # set base directory for eval artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # parse config file
    config_opts = parse_config(train.plugin_config)
    # parse extra deeplab args
    extra_opts = parse_deeplab_args(extra_deeplab_args)

    # union all options
    config_opts.update(extra_opts)

    # set up entry for "custom" dataset
    setup_dataset(train.dataset.path)

    # fill sys.argv to be passed to deeplab
    sys.argv = [sys.argv[0]]
    sys.argv.append("--dataset=custom")
    sys.argv.append(f"--dataset_dir={data_dir.absolute()}")
    sys.argv.append(f"--eval_logdir={artifact_dir.absolute()}")
    sys.argv.append(f"--checkpoint_dir={checkpoint_dir}")
    sys.argv.append("--eval_split=test")
    sys.argv += [f"--{key}={value}" for key, value in config_opts.items()]

    # run deeplab
    try:
        with TFRecordFilenameConverter(data_dir.absolute()):
            deeplab_eval.main(None)
    except KeyboardInterrupt:
        pass


@tf_semantic.command(help="Visualize a model. Always use in local mode.", context_settings=dict(ignore_unknown_options=True))
@pass_train
@click.argument("checkpoint_dir", type=click.Path(exists=True))
@click.argument('extra_deeplab_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def vis(ctx, train: TrainInput, checkpoint_dir, extra_deeplab_args):
    from deeplab import vis as deeplab_vis
    from deeplab import common as deeplab_common

    # stupid hack to prevent deeplab from skipping label reading when the split is called 'test'
    deeplab_common.TEST_SET = None

    # set base directory for eval artifacts
    artifact_dir = train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # parse config file
    config_opts = parse_config(train.plugin_config)
    # parse extra deeplab args
    extra_opts = parse_deeplab_args(extra_deeplab_args)

    # union all options
    config_opts.update(extra_opts)

    # set up entry for "custom" dataset
    setup_dataset(train.dataset.path)

    # fill sys.argv to be passed to deeplab
    sys.argv = [sys.argv[0]]
    sys.argv.append("--dataset=custom")
    sys.argv.append(f"--dataset_dir={data_dir.absolute()}")
    sys.argv.append(f"--vis_logdir={artifact_dir.absolute()}")
    sys.argv.append(f"--checkpoint_dir={checkpoint_dir}")
    sys.argv.append("--vis_split=test")
    sys.argv += [f"--{key}={value}" for key, value in config_opts.items()]

    # run deeplab
    try:
        with TFRecordFilenameConverter(data_dir.absolute()):
            deeplab_vis.main(None)
    except KeyboardInterrupt:
        pass

class TFRecordFilenameConverter:
    """
    Converts TFRecord files from the Jigsaw format to the deeplab format.
    """
    deeplab_format = "{splitname}-{index}-of-{total}.tfrecord"
    jigsaw_format = "{splitname}.record-{index}-of-{total}"
    jigsaw_regex = re.compile(r"^(?P<splitname>\w+)\.record-(?P<index>[0-9]+)-of-(?P<total>[0-9]+)$")

    def __init__(self, directory):
        self.directory = directory
        self.converted_file_dicts = []

    def __enter__(self):
        for filename in os.listdir(self.directory):
            match = re.fullmatch(self.jigsaw_regex, filename)
            if match:
                match_dict = match.groupdict()
                self.converted_file_dicts.append(match_dict)
                new_filename = self.deeplab_format.format(**match_dict)
                shutil.copyfile(os.path.join(self.directory, filename),
                                os.path.join(self.directory, new_filename))

    def __exit__(self, exc_type, exc_val, exc_tb):
        for match_dict in self.converted_file_dicts:
            os.remove(os.path.join(self.directory, self.deeplab_format.format(**match_dict)))
