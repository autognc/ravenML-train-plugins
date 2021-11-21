
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import click
import time
import tqdm
import cv2
import os
from pathlib import Path
from rmltraintfmobilepose import utils

from .helpers import (
    cull_models, 
    cull_error_metrics,
    cull_mask_generators,
)


def train_cullnet(
    model_path,
    directory,
    artifact_dir,
    object_name,
    stl_path,
    keypoints,
    focal_length,
    error_metric,
    mask_mode,
    model_type,
    eval_trained_cull_model,
    cache,
    experiment,
    hps
):
    extra_files = []
    model = tf.keras.models.load_model(model_path, compile=False)
    # print(model.summary())
    cache = Path(cache)
    cull_masks_paths = cache /"cull-masks.npy"
    cull_error_cache = cache /"cull-errors.npy"
    cull_true_rot_path = cache /"cull-true_rot_error.npy"
    keypoints_path = (
        keypoints if keypoints else os.path.join(directory, "keypoints.npy")
    )
    ref_points = np.load(keypoints_path).reshape((-1, 3))
    error_func, error_norm, error_denorm = cull_error_metrics[error_metric]

    if not os.path.exists(os.path.join(cache, "cull-masks.npy")):
        print("Using", model_path, "to generate cullnet training data...")
        mask_gen_init = cull_mask_generators[mask_mode]
        train_dir = directory / "splits" / "complete" / "train"
        mask_gen = mask_gen_init(stl_path, size=model.input.shape[1])
        X, y, true_rot_error = compute_model_error_training_data(
            model, train_dir, ref_points, focal_length, error_func, mask_gen, object_name
        )
        if cache:
            np.save(cull_masks_paths, X)
            np.save(cull_error_cache , y)
            np.save(cull_true_rot_path, true_rot_error)
    else:
        print("Using cached data...")
        X, y, true_rot_error = np.load(cull_masks_paths), np.load(cull_error_cache), np.load(cull_true_rot_path)
    print("Loaded dataset: ", X.shape, " -> ", y.shape)
    extra_files.append(keypoints_path)
    extra_files.append(cull_masks_paths)
    extra_files.append(cull_true_rot_path)
    extra_files.append(cull_error_cache)
    # patch inf weirdness
    y = np.clip(y, np.amin(y), np.nanmax(y[y != np.inf]))

    # Ensure error data/labels looks good to feed into the model
    ynorm = error_norm(y)
    print("error        mean=", y.mean(), "std=", y.std())
    print("error (norm) mean=", ynorm.mean(), "std=", ynorm.std())
    _, (ax, ax2) = plt.subplots(nrows=2)
    ax.hist(y)
    ax.set_title("error")
    ax2.hist(ynorm)
    ax2.set_title("error (norm)")
    plot_path = artifact_dir / "cull-error-dist.png"
    extra_files.append(plot_path)
    plt.savefig(plot_path)
    if experiment:
        experiment.log_image(plot_path)
    

    # Train/Test Shuffle & Split
    np.random.seed(0)
    shuffle_idxs = np.random.permutation(len(X))
    train_size = int(len(X) * 0.7)
    train_idxs, test_idxs = shuffle_idxs[:train_size], shuffle_idxs[train_size:]
    # these are all normalized
    X_train, y_train, X_test, y_test = (
        X[train_idxs],
        ynorm[train_idxs],
        X[test_idxs],
        ynorm[test_idxs],
    )

    if eval_trained_cull_model is None:
        print("Training cullnet...", X_train.shape, X_test.shape)
        model_name = model_type + "-" + str(int(time.time()))
        cull_model = cull_models[model_type](X.shape[1:], model)
        cull_model, cull_model_path = train_model(
            model_name, artifact_dir, cull_model, X_train, y_train, X_test, y_test, hps["batch_size"], hps["epochs"]
        )
    else:
        print("Using pretrained cullnet...", eval_trained_cull_model)
        model_name = (
            eval_trained_cull_model.replace("\\", "").replace("/", "").replace(":", "")
        )
        cull_model_path = eval_trained_cull_model
        cull_model = tf.keras.models.load_model(eval_trained_cull_model)
    if experiment:
        experiment.log_model(f"cullnet",str(cull_model_path))
    extra_files.append(cull_model_path)
    ynorm_pred = cull_model.predict(X).squeeze()
    ynorm_test_pred = cull_model.predict(X_test).squeeze()

    # Plot & Log results (includes results for all examples and just test examples, normalized and at original scale)
    _, ((ax, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    for data, pred_name, axis in zip(
        [
            (ynorm_pred, ynorm),
            (ynorm_test_pred, y_test),
            (error_denorm(ynorm_pred), error_denorm(ynorm)),
            (error_denorm(ynorm_test_pred), error_denorm(y_test)),
        ],
        ["y (norm)", "y_test (norm)", "y", "y_test"],
        [ax, ax2, ax3, ax4],
    ):
        l2_pred = np.linalg.norm(data[0] - data[1])
        corr_pred = np.corrcoef(data[0], data[1])[0, 1]
        print(pred_name, "l2=", l2_pred, "r=", corr_pred)
        axis.scatter(data[0], data[1])
        axis.set_title(pred_name)
    plot_path = artifact_dir / ("cull-" + model_name + "-predictions.png")
    extra_files.append(plot_path)
    plt.savefig(plot_path)
    if experiment:
        experiment.log_image(plot_path)

    # er curve
    plt.clf()
    y = np.cumsum(np.degrees(true_rot_error)[np.argsort(ynorm_pred)]) / (np.arange(len(true_rot_error)) + 1)
    x = (np.arange(len(ynorm_pred)) + 1) / len(ynorm_pred)
    fig, ax1 = plt.subplots()
    ax1.scatter(x, y)
    ax1.set_xlabel("proportion of images kept")
    ax1.set_ylabel("mean error", color="blue")
    ax2 = ax1.twinx()
    ax2.scatter(x, np.sort(ynorm_pred)[::-1], color="orange")
    ax2.set_ylabel("confidence threshold", color="orange")
    plot_path = artifact_dir / "er_curve.png"
    extra_files.append(plot_path)
    plt.savefig(plot_path)
    if experiment:
        experiment.log_image(plot_path)
    np.save(artifact_dir / 'true_rot_error-ynorm-ynorm_pred.npy', (true_rot_error, ynorm, ynorm_pred))
    extra_files.append(artifact_dir / 'true_rot_error-ynorm-ynorm_pred.npy')
    extra_files = [Path(f) for f in extra_files]
    return Path(cull_model_path), extra_files

def train_model(model_name, artifact_dir, model, X_train, y_train, X_test, y_test, batch_size, epochs):
    save_name = artifact_dir / (model_name + "-best.h5")
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
        # epochs is set high, use a *single* ^C to finish training
        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("Stopping...")
    print("Loading best save...")
    return tf.keras.models.load_model(save_name), save_name


def derive_keypoints(truth, ref_points):
    r_vec = utils.pose.to_rotation(truth["pose"]).as_rotvec()
    t_vec = truth["position"].numpy()
    imdims = truth["imdims"]
    focal_length = truth["focal_length"]
    cam_matrix = np.array(
        [
            [focal_length, 0, imdims[1] // 2],
            [0, focal_length, imdims[0] // 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    kps = cv2.projectPoints(ref_points, r_vec, t_vec, cam_matrix, None)[0][
        :, :, ::-1
    ].squeeze()
    return kps

def compute_model_error_training_data(
    model, directory, ref_points, default_focal_length, error_func, mask_gen, object_name
):
    nb_keypoints = model.output.shape[-1] // 2
    cropsize = model.input.shape[1]
    ref_points = ref_points[:nb_keypoints]

    data = utils.data.dataset_from_directory(
        directory,
        cropsize,
        nb_keypoints=nb_keypoints,
        focal_length=default_focal_length,
        object_name=object_name
    )
    data = data.batch(32)

    inputs = []
    outputs = []
    true_rot_error = []

    for image_batch, truth_batch in tqdm.tqdm(data):
        kps_batch = model.predict(image_batch)
        kps_batch = utils.model.decode_displacement_field(kps_batch)
        kps_batch_cropped = kps_batch * (cropsize // 2) + (cropsize // 2)
        kps_batch_uncropped = (
            kps_batch
            / 2
            * truth_batch["bbox_size"][
                :,
                None,
                None,
                None,
            ]
            + truth_batch["centroid"][:, None, None, :]
        )
        truth_batch = [
            dict(zip(truth_batch.keys(), t)) for t in zip(*truth_batch.values())
        ]
        for image, kps_cropped, kps_uncropped, truth in zip(
            image_batch.numpy(),
            kps_batch_cropped.numpy(),
            kps_batch_uncropped.numpy(),
            truth_batch,
        ):
            # get pose solution in uncropped reference frame
            r_vec, t_vec = utils.pose.solve_pose(
                ref_points,
                kps_uncropped,
                [truth["focal_length"], truth["focal_length"]],
                truth["imdims"],
                ransac=True,
                reduce_mean=False,
            )

            extra_crop_params = {
                "centroid": truth["centroid"],
                "bbox_size": truth["bbox_size"],
                "imdims": truth["imdims"],
            }
            img_mask = mask_gen.make_and_apply_mask(
                image, r_vec, t_vec, truth["focal_length"], extra_crop_params
            )
            inputs.append(img_mask)
            cv2.imwrite("trash/mask.png",img_mask*255)
            if "keypoints" not in truth:
                unscaled_kps = derive_keypoints(truth, ref_points)
            else:
                unscaled_kps = truth["keypoints"]
            # L2 keypoint error is computed in cropped reference frame
            kps_true = (unscaled_kps - truth["centroid"]) / truth[
                "bbox_size"
            ] * cropsize + (cropsize // 2)
            error = error_func(
                kps_cropped, kps_true, r_vec, t_vec, truth["pose"], truth["position"]
            )
            outputs.append(error)
            true_rot_error.append(utils.pose.geodesic_error(r_vec, truth["pose"]))
    return np.array(inputs), np.array(outputs), np.array(true_rot_error)
