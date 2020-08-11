import tensorflow as tf
import click
from .. import utils

help_string = """
Freeze a model from the Keras .h5 checkpoint to the Tensorflow SavedModel format so that it
can be easily served from anywhere. Requires Tensorflow 2.0.

The resulting saved model will take a Tensor([cropsize, cropsize, 3], dtype=uint8) as input
and output a Tensor([4], dtype=float32) pose prediction.

The resulting saved model can be loaded for inference using TF 1.x or 2.x using the following code:
    model = tf.compat.v2.saved_model.load(path)
The input shape can then be retrieved with:
    input_shape = model.__call__.concrete_functions[0].inputs[0].shape

"""


@click.command(help=help_string)
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(exists=False, file_okay=False))
def main(model_path, output_path):
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"tf": tf}
    )

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

        @tf.function(
            input_signature=[tf.TensorSpec(model.input.shape[1:], dtype=tf.uint8)]
        )
        def __call__(self, image):
            batch_im = tf.expand_dims(image, 0)
            normalized = tf.keras.applications.mobilenet_v2.preprocess_input(
                tf.cast(batch_im, tf.float32)
            )
            df = model(normalized)
            kps_batch = utils.model.decode_displacement_field(df)
            kps_batch = kps_batch * (self.cropsize // 2) + (self.cropsize // 2)
            return tf.squeeze(kps_batch)

    module = Module(model)
    tf.saved_model.save(module, output_path)
