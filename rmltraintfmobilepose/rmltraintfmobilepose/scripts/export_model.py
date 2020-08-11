"""
Freeze a model from the Keras .h5 checkpoint to the Tensorflow SavedModel format so that it
can be easily served from anywhere. Requires Tensorflow 2.0.

The resulting saved model will take a Tensor([cropsize, cropsize, 3], dtype=uint8) as input
and output a Tensor([4], dtype=float32) pose prediction.

The resulting saved model can be loaded for inference using TF 1.x or 2.x using the following code:
    model = tf.compat.v2.saved_model.load(path)
The input shape can then be retrieved with:
    input_shape = model.__call__.concrete_functions[0].inputs[0].shape

"""

import tensorflow as tf
import argparse
import numpy as np
from rmltraintfkeypoints.train import KeypointsModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Path to Keras .h5 file")
    parser.add_argument('output', type=str, help="Output path")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False, custom_objects={'tf': tf})

    # for some reason, directly trying to save the keras model isn't working,
    # so let's create a custom one as a workaround. This conveniently also
    # allows us to build in the preprocessing and get rid of the batch dimension
    class Module(tf.Module):
        cropsize = model.input.shape[1]
        nb_keypoints = model.output_shape[-1] // 2
        dfdims = model.output_shape[1:3]

        def __init__(self, model):
            # directly copy over the variables to get rid of keras weirdness
            for layer in model.layers:
                for variable in layer.variables:
                    setattr(self, variable.name, variable)

        @tf.function(input_signature=[tf.TensorSpec(model.input.shape[1:], dtype=tf.uint8)])
        def __call__(self, image):
            batch_im = tf.expand_dims(image, 0)
            normalized = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(batch_im, tf.float32))
            df = model(normalized)
            kps_batch = KeypointsModel.decode_displacement_field(df)
            kps_batch = tf.transpose(kps_batch, [0, 3, 1, 2])
            # if using the reduce_mean strategy, comment out the next line
            # and pass ransac=False to calculate_pose_vectors.
            kps_batch = tf.reshape(kps_batch, [tf.shape(kps_batch)[0], -1, 2])
            # kps_batch = tf.reduce_mean(kps_batch, axis=1)
            kps_batch = kps_batch * (self.cropsize // 2) + (self.cropsize // 2)
            return tf.squeeze(kps_batch)

    module = Module(model)
    tf.saved_model.save(module, args.output)


if __name__ == "__main__":
    main()
