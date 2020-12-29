"""
Train and/or eval CullNet

Show Help
$ ravenml train --config train_config.json tf-mobilepose cull --help

Train a `mobilenetv2_imagenet_mse` cullnet model using `numpy_cut`-masks. Use `cull.npy` to cache error data.
$ ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test 
    -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse

Eval model `mobilenetv2_imagenet_mse-1609270326-best.h5` on `~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test` data.
$ ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test 
    -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse -t mobilenetv2_imagenet_mse-1609270326-best.h5
"""
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import click
import time
import tqdm
import cv2
import os

from .. import utils


class MaskGenerator:
    def __init__(self, size=224, stack=True):
        self.stack = stack
        self.size = size

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        raise NotImplementedError()

    def make_and_apply_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        mask = self.make_binary_mask(image, r_vec, t_vec, focal_length, extra_crop_params)
        assert mask.shape[:2] == image.shape[:2]
        if self.stack:
            w, h, c = image.shape
            assert c == 3
            image_and_mask = np.empty((w, h, c + 1))
            image_and_mask[:, :, :3] = image
            # mask is [0, 1]
            image_and_mask[:, :, 3] = mask * 1
        else:
            image_and_mask = image.copy()
            # -1 dependent on how image is encoded
            image_and_mask[np.where(mask == 0)] = [-1, -1, -1]
        return image_and_mask


class NumpyMaskProjector(MaskGenerator):
    def __init__(self, all_model_keypoints, dilate_iters=2, **kwargs):
        super().__init__(**kwargs)
        self.dilate_iters = dilate_iters
        self.all_kps_homo = np.hstack([all_model_keypoints, np.ones((len(all_model_keypoints), 1))]).T

    def make_binary_mask(self, image, r_vec, t_vec, focal_length, extra_crop_params):
        imdims = np.array(image.shape[:2])
        original_imdims = np.array(extra_crop_params["imdims"])
        origin = (
            np.array(extra_crop_params["centroid"]) - extra_crop_params["bbox_size"] / 2
        )
        center = original_imdims / 2 - origin
        focal_length *= imdims / extra_crop_params["bbox_size"]
        center *= imdims / extra_crop_params["bbox_size"]
        cam_matrix = np.array(
            [[focal_length[1], 0, center[1]], [0, focal_length[0], center[0]], [0, 0, 1]],
            dtype=np.float32,
        )
        rot_matrix = Rotation.from_rotvec(r_vec.reshape((3,))).as_matrix()
        proj = cam_matrix @ np.hstack([rot_matrix, t_vec]) @ self.all_kps_homo
        coords = ((proj / proj[2])[:2].T).astype(np.uint8)
        coords = np.clip(coords, 0, self.size - 1)
        img = np.zeros((self.size, self.size))
        img[coords[:, 1], coords[:, 0]] = 1
        img = cv2.dilate(img, (4, 4), iterations=self.dilate_iters)
        return img


def create_model_mobilenetv2_imagenet_mse(input_shape):
    model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=True,
        weights="imagenet",
        # These are ignored, but required for weights
        classes=1000,
        classifier_activation="softmax",
    )
    new_input = model.input
    feat_out = model.layers[-2].output
    out = tf.keras.layers.Dense(1, activation="linear")(feat_out)
    full_model = tf.keras.models.Model(new_input, out)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=[])
    return full_model


# Several possible cullnet models
#   name: model_generator_function
cull_models = {
    "mobilenetv2_imagenet_mse": create_model_mobilenetv2_imagenet_mse
}


# Several possible error metrics
#   name: (error_calc_function, normalize, denormalize)
cull_error_metrics = {
    "keypoint_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true: 
            np.mean([np.linalg.norm(kp_true - kp) for kp, kp_true in zip(kps_pred, kps_true)]),
        lambda y: np.clip((np.log(y) - 5.935) / 0.1731, -3, 3),
        lambda ynorm: np.exp(ynorm * 0.1731 + 5.935)),
    "geodesic_rotation": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true:
            utils.pose.geodesic_error(r_vec, pose_true),
        lambda y: y, # TODO
        lambda ynorm: ynorm),
    "position_l2": (
        lambda kps_pred, kps_true, r_vec, t_vec, pose_true, position_true:
            utils.pose.position_error(t_vec, position_true)[0],
        lambda y: y, # TODO
        lambda ynorm: ynorm)
}


# Several possible mask encodings
#   name: constructor
# "_stack" will stack the mask on to the image as another channel
#       this is what's done in the original cullnet paper
# "_cut" (passed as stack=False) will cut out the shape of the mask from the original image
#       this is cool b/c it allows one to reuse 3-channel trained models
cull_mask_generators = {
    "numpy_stack": lambda *args, **kwargs: NumpyMaskProjector(*args, **kwargs, stack=True),
    "numpy_cut": lambda *args, **kwargs: NumpyMaskProjector(*args, **kwargs, stack=False),
}


@click.command(help="Train/Use CullNet")
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-k",
    "--keypoints",
    help='Path to 3D reference points .npy file (optional). Defaults to "{directory}/keypoints.npy"',
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("-f", "--focal_length", type=float, required=True)
@click.option(
    "-e", 
    "--error_metric", 
    type=click.Choice(cull_error_metrics.keys(), case_sensitive=False), 
    default=next(iter(cull_error_metrics))
)
@click.option(
    "-k", 
    "--mask_mode", 
    type=click.Choice(cull_mask_generators.keys(), case_sensitive=False), 
    default=next(iter(cull_mask_generators))
)
@click.option(
    "-m", 
    "--model_type", 
    type=click.Choice(cull_models.keys(), case_sensitive=False), 
    default=next(iter(cull_models))
)
@click.option("-t", "--eval_trained_cull_model", type=click.Path(exists=True, dir_okay=False))
@click.option("-c", "--cache", help="Cache dataset here", type=click.Path(dir_okay=False))
def main(model_path, directory, keypoints, focal_length, error_metric, mask_mode, model_type, eval_trained_cull_model, cache):

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(), loss=lambda _: 0,
    )
    error_func, error_norm, error_denorm = cull_error_metrics[error_metric]

    if not cache or not os.path.exists('cull-masks.' + cache):
        # Use pose model to make cull training data
        mask_gen_init = cull_mask_generators[mask_mode]
        # TODO make hardcoded stuff into options
        if mask_mode.startswith("numpy"):
            mask_gen = mask_gen_init(np.load("keypoints700k.npy"), size=224)
        else:
            mask_gen = mask_gen_init(size=224)
        X, y = compute_model_error_training_data(model, directory, keypoints, focal_length, error_func, mask_gen)
        if cache:
            np.save('cull-masks.' + cache, X)
            np.save('cull-errors.' + cache, y)
    else:
        print('Using cached data...')
        X, y = np.load('cull-masks.' + cache), np.load('cull-errors.' + cache)
    print('Loaded dataset: ', X.shape, ' -> ', y.shape)

    ynorm = error_norm(y)
    # Ensure error data/labels looks good to feed into the model
    print('error        mean=', y.mean(), 'std=', y.std())
    print('error (norm) mean=', ynorm.mean(), 'std=', ynorm.std())
    _, (ax, ax2) = plt.subplots(nrows=2)
    ax.hist(y)
    ax.set_title('error')
    ax2.hist(ynorm)
    ax2.set_title('error (norm)')
    plt.savefig('cull-errors.png')

    # Train/Test Shuffle & Split
    np.random.seed(0)
    shuffle_idxs = np.random.permutation(len(X))
    train_size = int(len(X) * 0.7)
    train_idxs, test_idxs = shuffle_idxs[:train_size], shuffle_idxs[train_size:]
    # these are all normalized
    X_train, y_train, X_test, y_test = X[train_idxs], ynorm[train_idxs], X[test_idxs], ynorm[test_idxs]

    if eval_trained_cull_model is None:
        print('Training cullnet...')
        model_name = model_type + "-" + str(int(time.time()))
        cull_model = cull_models[model_type](X.shape[1:])
        cull_model = train_model(model_name, cull_model, X_train, y_train, X_test, y_test)
    else:
        print('Using pretrained cullnet...')
        model_name = eval_trained_cull_model.replace('\\', '').replace('/', '').replace(':', '')
        cull_model = tf.keras.models.load_model(eval_trained_cull_model)

    ynorm_pred = cull_model.predict(X).squeeze()
    ynorm_test_pred = cull_model.predict(X_test).squeeze()

    # Plot & Log results (includes results for all examples and just test examples, normalized and at original scale)
    _, ((ax, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for data, pred_name, axis in zip(
        [(ynorm_pred, ynorm), (ynorm_test_pred, y_test), (error_denorm(ynorm_pred), error_denorm(ynorm)), (error_denorm(ynorm_test_pred), error_denorm(y_test))], 
        ['y (norm)', 'y_test (norm)', 'y', 'y_test'], 
        [ax, ax2, ax3, ax4]
    ):
        l2_pred = np.linalg.norm(data[0] - data[1])
        corr_pred = np.corrcoef(data[0], data[1])[0, 1]
        print(pred_name, 'l2=', l2_pred, 'r=', corr_pred)
        axis.scatter(data[0], data[1])
        axis.set_title(pred_name)
    plt.savefig(model_name + '-predictions.png')


def train_model(model_name, model, X_train, y_train, X_test, y_test):
    save_name = model_name + "-best.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_name,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=True,
        )
    ]
    try:
        # epochs is set high, use ^C to finish training
        model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=500,
            verbose=2,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print('Stopping...')
    print('Loading best save...')
    return tf.keras.models.load_model(save_name)


def compute_model_error_training_data(model, directory, keypoints, focal_length, error_func, mask_gen):

    nb_keypoints = model.output.shape[-1] // 2
    cropsize = model.input.shape[1]

    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        directory, cropsize, nb_keypoints=nb_keypoints
    )
    data = data.batch(32)

    inputs = []
    outputs = []

    for image_batch, truth_batch in tqdm.tqdm(data):
        truth_batch = [
            dict(zip(truth_batch.keys(), t)) for t in zip(*truth_batch.values())
        ]
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch = kps_batch * (cropsize // 2) + (cropsize // 2)
        for image, kps, truth in zip(image_batch, kps_batch.numpy(), truth_batch):
            image = image.numpy()
            kps_true = (truth["keypoints"] - truth["centroid"]) / truth[
                "bbox_size"
            ] * cropsize + (cropsize // 2)
            extra_crop_params = {
                "centroid": truth["centroid"],
                "bbox_size": truth["bbox_size"],
                "imdims": truth["imdims"],
            }
            r_vec, t_vec = utils.pose.solve_pose(
                ref_points,
                kps,
                [focal_length, focal_length],
                image.shape[:2],
                extra_crop_params=extra_crop_params,
                ransac=True,
                reduce_mean=False,
            )
            img_mask = mask_gen.make_and_apply_mask(image, r_vec, t_vec, focal_length, extra_crop_params)
            inputs.append(img_mask)
            error = error_func(kps, kps_true, r_vec, t_vec, truth["pose"], truth["position"])
            outputs.append(error)
    return np.array(inputs), np.array(outputs)
