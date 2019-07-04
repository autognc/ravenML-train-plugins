import tensorflow as tf
import numpy as np
import os
import sys
import math
import glob
from tqdm import tqdm
from deeplab.datasets import build_data
from PIL import Image

_NUM_SHARDS = 4


def to_tfrecord(image_dir, label_dir, split_name, output_dir):
    """Converts the data from grayscale into into tfrecord format.

    Args:
      image_dir: dir with image data
      label_dir: dir with label data
      output_dir: output directory
      split_name: train, val, test, etc

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """
    tf.gfile.MakeDirs(output_dir)

    img_names = tf.gfile.Glob(os.path.join(image_dir, '*.png'))
    seg_names = []
    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(label_dir, basename + '.png')
        seg_names.append(seg)

    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            output_dir,
            '%s-%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = img_names[i]
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = seg_names[i]
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

def to_grayscale(input_dir, output_dir):
    """
    Converts the data from our RGB 3-channel color format to deeplab's expected 1-channel format,
    where each class is a consecutive grayscale value. Writes a CSV file to output_dir with mapping used.
    """
    tf.gfile.MakeDirs(output_dir)
    img_names = glob.glob(os.path.join(input_dir, "*.png"))

    # detect unique colors for classes
    colors = set()
    for fn in tqdm(img_names, "Detecting classes"):
        img = Image.open(fn)
        c = {t for _, t in img.getcolors(256)}
        colors = colors.union(c)

    print(f"{len(colors)} classes (unique colors) detected:")
    print(colors)

    colors = np.array(list(colors))
    classes = np.arange(len(colors))
    # write out CSV file with luminance to RGB mapping
    with open(os.path.join(output_dir, "colormap.csv"), "w") as csvfile:
        csvfile.write("L,R,G,B\n")
        for i, color in zip(classes, colors):
            csvfile.write(str(i))
            csvfile.write(",")
            csvfile.write(",".join(map(str, color)))
            csvfile.write("\n")

    for fn in tqdm(img_names, "Transforming to grayscale"):
        img = np.array(Image.open(fn))
        # numpy broadcasting magic
        masks = (img[..., None] == colors.T).all(axis=2)
        result = np.where(masks, classes, 0).sum(axis=2)
        Image.fromarray(result.astype(np.uint8)).save(os.path.join(output_dir, os.path.basename(fn)))
