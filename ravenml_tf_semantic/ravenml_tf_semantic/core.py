import click
import yaml
from ravenml.train.options import pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from datetime import datetime
import glob
import sys
import re
import os
import shutil

from ravenml.utils.local_cache import LocalCache, global_cache


@click.group(help='TensorFlow Semantic Segmentation.')
def tf_semantic():
    pass


@tf_semantic.command(help="Train a model.", context_settings=dict(ignore_unknown_options=True))
@pass_train
@click.option("--config", "-c", required=False, type=click.Path(exists=True),
              help="Config file containing command-line parameters to deeplab/train.py.")
@click.argument('extra_deeplab_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def train(ctx, train: TrainInput, config, extra_deeplab_args):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train".
    # After training, create an instance of TrainOutput and return it
    from deeplab.datasets import data_generator
    from deeplab import train as deeplab_train

    # set base directory for model artifacts
    artifact_dir = LocalCache(global_cache.path / 'tf-semantic-deeplab') if train.artifact_path is None \
        else train.artifact_path

    # set dataset directory
    data_dir = train.dataset.path / "splits" / "complete" / "train"

    # parse config file
    config_opts = {}
    if config is not None:
        with open(config, 'r') as f:
            try:
                args = yaml.safe_load(f)
            except yaml.YAMLError as e:
                ctx.fail(str(e))
        for arg in args:
            arg = list(arg.items())
            if len(arg) > 1:
                ctx.fail("Invalid config file, please only specify one key-value pair per list item")
            config_opts[arg[0][0]] = str(arg[0][1])

    # parse extra deeplab args
    extra_opts = {}
    for arg in extra_deeplab_args:
        split = extra_deeplab_args[2:].split("=")
        if len(split) != 2:
            ctx.fail(f"Improperly formatted deeplab arg {arg}")
        extra_opts[split[0]] = split[1]

    # union all options
    config_opts.update(extra_opts)

    # read number of classes from Jigsaw label map, this is bad but I'm not gonna pull out
    # a protobuf parser just to count the number of IDs
    with open(train.dataset.path / "label_map.pbtxt", "r") as f:
        ids = [line for line in f if "id:" in line]
        num_classes = len(ids)

    # set up data generator for our dataset
    dataset_info = data_generator.DatasetDescriptor(
        splits_to_sizes={
            'train': -1,  # these aren't actually used
            'val': -1,
        },
        num_classes=num_classes + 1,
        ignore_label=0,
    )
    data_generator._DATASETS_INFORMATION['custom'] = dataset_info

    # fill metadata
    metadata = {
        'date_started_at': datetime.utcnow().isoformat() + "Z",
        'dataset_used': train.dataset.metadata,
        'num_classes': num_classes,
        'deeplab_options': config_opts,
    }

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
    checkpoint_files = glob.glob(str(artifact_dir.absolute() / "*"))
    return TrainOutput(metadata, artifact_dir, model_path, checkpoint_files, train.artifact_path is not None)


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
