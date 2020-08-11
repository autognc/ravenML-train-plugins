import tensorflow as tf


def preprocess_image(image_data, centroid, bbox_size, cropsize):
    """
    Performs preproccessing on a single image for feeding into the network.
    :param image_data: raw image data as a bytestring
    :param centroid: the center of the bounding box to crop to, in pixel coordinates
    :param bbox_size: the side length of the bbox to crop to, in pixels
    :param cropsize: the output size of the cropped image
    :return: the decoded image cropped to a [bbox_size, bbox_size] square centered around centroid
    and resized to [cropsize, cropsize]
    """
    image = tf.io.decode_image(image_data, channels=3)
    # this rescales inputs to the range [-1, 1], which should be what the model expects
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(image, tf.float32)
    )

    # ensure types
    bbox_size = tf.cast(bbox_size, tf.float32)
    centroid = tf.cast(centroid, tf.float32)

    # convert to [0, 1] relative coordinates
    imdims = tf.cast(tf.shape(image)[:2], tf.float32)
    centroid /= imdims
    bbox_size /= imdims  # will broadcast to shape [2]

    # crop to (bbox_size, bbox_size) centered around centroid and resize to (cropsize, cropsize)
    bbox_size /= 2
    image = tf.squeeze(
        tf.image.crop_and_resize(
            tf.expand_dims(image, 0),
            [
                [
                    centroid[0] - bbox_size[0],
                    centroid[1] - bbox_size[1],
                    centroid[0] + bbox_size[0],
                    centroid[1] + bbox_size[1],
                ]
            ],
            [0],
            [cropsize, cropsize],
            extrapolation_value=-1,
        )
    )
    image = tf.ensure_shape(image, [cropsize, cropsize, 3])
    return imdims, image


def encode_displacement_field(keypoints, dfdims):
    """
    :param keypoints: a shape (b, n, 2) Tensor with N keypoints normalized to (-1, 1)
    :param dfdims: a shape [2] Tensor with the dimensions of the displacement field
    :return: a shape (b, height, width, 2n) Tensor
    """
    delta = 2 / tf.convert_to_tensor(dfdims, dtype=tf.float32)
    y_range = tf.range(-1, 1, delta[0]) + (delta[0] / 2)
    x_range = tf.range(-1, 1, delta[1]) + (delta[1] / 2)
    mgrid = tf.stack(
        tf.meshgrid(y_range, x_range, indexing="ij"), axis=-1
    )  # shape (y, x, 2)
    df = keypoints[:, :, None, None, :] - mgrid  # shape (b, n, y, x, 2)
    df = tf.transpose(df, [0, 2, 3, 1, 4])  # shape (b, y, x, n, 2)
    return tf.reshape(df, [tf.shape(keypoints)[0], dfdims[0], dfdims[1], -1])


def decode_displacement_field(df):
    """
    :param df: a shape (b, height, width, 2n) displacement field
    :return: a shape (b, height * width, n, 2) tensor where each keypoint has height * width predictions
    """
    dfdims = tf.shape(df)[1:3]
    df = tf.reshape(
        df, [tf.shape(df)[0], dfdims[0], dfdims[1], -1, 2]
    )  # shape (b, y, x, n, 2)
    delta = tf.cast(2 / dfdims, tf.float32)
    y_range = tf.range(-1, 1, delta[0]) + (delta[0] / 2)
    x_range = tf.range(-1, 1, delta[1]) + (delta[1] / 2)
    mgrid = tf.stack(
        tf.meshgrid(y_range, x_range, indexing="ij"), axis=-1
    )  # shape (y, x, 2)
    keypoints = df + mgrid[:, :, None, :]  # shape (b, y, x, n, 2)
    return tf.reshape(
        keypoints, [tf.shape(df)[0], dfdims[0] * dfdims[1], -1, 2]
    )  # shape (b, y*x, n, 2)


def preprocess_keypoints(parsed_kps, centroid, bbox_size, img_size, nb_keypoints):
    """Normalizes keypoints to the [-1, 1] range for training"""
    keypoints = tf.reshape(parsed_kps, [-1, 2])[:nb_keypoints]
    keypoints *= img_size
    keypoints = (keypoints - centroid) / (bbox_size / 2)
    return tf.reshape(keypoints, [nb_keypoints * 2])
