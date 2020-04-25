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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help="Path to Keras .h5 file", required=True)
    parser.add_argument('-o', '--output', type=str, help="Output path", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False, custom_objects={'tf': tf})

    # for some reason, directly trying to save the keras model isn't working,
    # so let's create a custom one as a workaround. This conveniently also
    # allows us to build in the preprocessing and get rid of the batch dimension
    class Module(tf.Module):
        def __init__(self, model):
            # directly copy over the variables to get rid of keras weirdness
            for layer in model.layers:
                for variable in layer.variables:
                    setattr(self, variable.name, variable)

        @tf.function(input_signature=[tf.TensorSpec(model.input.shape[1:], dtype=np.uint8)])
        def __call__(self, image):
            batch_im = tf.expand_dims(image, 0)
            normalized = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(batch_im, tf.float32))
            return tf.squeeze(model(normalized))

    module = Module(model)
    tf.saved_model.save(module, args.output)


if __name__ == "__main__":
    main()
