from distutils.log import error
import tensorflow as tf
import numpy as np
import traceback
import os
import cv2
#from tensorflow.python.keras.applications.mobilenet_v3 import _inverted_res_block, hard_swish
from tensorflow.python.keras.applications.mobilenet_v2 import _inverted_res_block

#TODO: dont pass in hp, check to see what's stored in hyperparameter object, if it's msotly irrelevant just pass in necessary values
# be more descriptive with names

#file contains all training model architectures. 

#mobilenet v2 architecture
def mbnv2_gen(hp):
    init_weights = hp["model_init_weights"]
    assert init_weights in ["imagenet", ""]
    mobilenet = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=init_weights if init_weights != "" else None,
        input_shape=(hp["crop_size"], hp["crop_size"], 3),
        pooling=None,
        alpha=1.0,
    )

    x = mobilenet.get_layer("block_16_project_BN").output

    # 7x7x160 -> 14x14x96
    x = tf.keras.layers.Conv2DTranspose(
        filters=96, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.concatenate([x, mobilenet.get_layer("block_12_add").output])
    x = _inverted_res_block(
        x, filters=96, alpha=1.0, stride=1, expansion=6, block_id=17
    )
    x = _inverted_res_block(
        x, filters=96, alpha=1.0, stride=1, expansion=6, block_id=18
    )

    x = tf.keras.layers.SpatialDropout2D(hp["dropout"])(x)
    # output 1x1 conv
    x = tf.keras.layers.Conv2D(hp["keypoints"] * 2, kernel_size=1, use_bias=True)(x)
    return tf.keras.models.Model(mobilenet.input, x, name="mobilepose")

#mobilenet v3 architecture
def mbnv3_gen(hp):
    init_weights = hp["model_init_weights"]
    assert init_weights in ["imagenet", ""]
    mobilenet = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights=init_weights,
        input_shape=(hp["crop_size"], hp["crop_size"], 3),
        pooling=None,
    )
    x = mobilenet.get_layer("expanded_conv_14/Add").output

    # 7x7x160 -> 14x14x112
    x = tf.keras.layers.Conv2DTranspose(
        filters=112, kernel_size=3, strides=2, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.concatenate([x, mobilenet.get_layer("expanded_conv_11/Add").output])
    
    x = _inverted_res_block(
        x, filters=112, kernel_size=3, stride=1, expansion=6, activation=hard_swish, se_ratio=0.25, block_id=17
    )
    x = _inverted_res_block(
        x, filters=112, kernel_size=3, stride=1, expansion=6, activation=hard_swish, se_ratio=0.25, block_id=18
    )

    x = tf.keras.layers.SpatialDropout2D(hp["dropout"])(x)

    # output 1x1 conv
    #TODO figure out self.nb_keypoints
    x = tf.keras.layers.Conv2D(hp["keypoints"] * 2, kernel_size=1, use_bias=True)(x)
    return tf.keras.models.Model(mobilenet.input, x, name="mobilepose")
    
def get_model_dict(model_arch_name):
        MODEL_ARCHITECTURES = {"mbnv2": mbnv2_gen,"mbnv3": mbnv3_gen}
        fn_gen_name = MODEL_ARCHITECTURES[model_arch_name]
        if fn_gen_name is not None:
            return fn_gen_name
        return error

#hp = hyperparameters from train_config
def create_model_from_config(hp):
    #training model for mobilenetv2 architectures
    return get_model_dict(hp["model_architecture"])(hp)