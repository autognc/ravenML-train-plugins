import click
from ravenml.train.options import kfold_opt, pass_train
from ravenml.train.interfaces import TrainInput, TrainOutput
from ravenml_tf_semantic.data_tools import to_tfrecord, to_grayscale
import os
import sys

from deeplab.datasets import data_generator
from deeplab import train as deeplab_train


@click.group(help='TensorFlow Semantic Segmentation.')
@click.pass_context
def tf_semantic(ctx):
    pass


@tf_semantic.command(help="Train a model.", context_settings=dict(ignore_unknown_options=True))
@pass_train
@click.option("--config", "-c", required=False, type=click.Path(exists=True),
              help="Config file containing command-line parameters to deeplab/train.py.\
              Colon-separated arguments, one per line.")
@click.argument('extra_deeplab_args', nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def train(ctx, train: TrainInput, config, extra_deeplab_args):
    # If the context has a TrainInput already, it is passed as "train"
    # If it does not, the constructor is called AUTOMATICALLY
    # by Click because the @pass_train decorator is set to ensure
    # object creation, after which the created object is passed as "train"
    # after training, create an instance of TrainOutput and return it

    # parse config file
    config_args = []
    if config is not None:
        with open(config, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                line = line.split(":")
                if len(line) != 2:
                    raise RuntimeError(f"Malformatted config file {config}")
                config_args.append(f"--{line[0].strip()}={line[1].strip()}")

    # set up data generator for our dataset
    dataset_info = data_generator.DatasetDescriptor(
        splits_to_sizes={
            'train': -1,  # these aren't actually used
            'val': -1,
        },
        num_classes=6,  # TODO: read from Jigsaw metadata
        ignore_label=255,
    )
    data_generator._DATASETS_INFORMATION['custom'] = dataset_info

    sys.argv = [sys.argv[0]]
    sys.argv.append("--dataset=custom")
    sys.argv.append(f"--dataset_dir={str(train.dataset.path.absolute()) + '/tfrecord'}")
    sys.argv.append(f"--train_logdir={str(train.artifact_path.absolute())}")
    sys.argv.append("--initialize_last_layer=False")
    sys.argv.append("--last_layers_contain_logits_only=True")
    sys.argv += config_args
    sys.argv += list(extra_deeplab_args)

    deeplab_train.main(None)
    return TrainOutput({'architecture': 'deeplab'}, train.artifact_path, train.artifact_path, [], True)


# TODO: move this to Jigsaw (in progress)
@tf_semantic.command(help="Convert a dataset to tfrecord format.")
@click.argument("input_dirs", nargs=-1)
@click.argument("output_dir")
@click.option("--image_subdir_name", default="real")
@click.option("--mask_subdir_name", default="mask")
@click.option("--mode", type=click.Choice(["color_to_grayscale", "grayscale_to_tfrecord", "both"]), default="both")
@click.pass_context
def convert(ctx, input_dirs, output_dir, image_subdir_name, mask_subdir_name, mode):
    # prevent command-line flags from being stupidly passed on to deeplab code
    sys.argv = [sys.argv[0]]

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    if mode != "grayscale_to_tfrecord":
        grayscale_dir = os.path.join(output_dir, "grayscale_masks")
        try:
            os.mkdir(grayscale_dir)
        except FileExistsError:
            pass

    for dir in input_dirs:
        split_name = os.path.split(dir)[-1]
        print(f"Processing {dir}, split name '{split_name}'")
        images_dir = os.path.join(dir, image_subdir_name)
        masks_dir = os.path.join(dir, mask_subdir_name)
        if mode == "color_to_grayscale":
            to_grayscale(masks_dir, grayscale_dir)
        elif mode == "grayscale_to_tfrecord":
            to_tfrecord(images_dir, masks_dir, split_name, output_dir)
        else:
            to_grayscale(masks_dir, grayscale_dir)
            to_tfrecord(images_dir, grayscale_dir, split_name, output_dir)
