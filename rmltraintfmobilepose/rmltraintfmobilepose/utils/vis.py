import numpy as np
import cv2


def vis_keypoints(image, kps, err_rot=None, err_pos=None):
    """
    :param kps: shape (guesses_per_kp, nb_keypoints, 2)
    """
    nb_keypoints = kps.shape[1]
    kps_by_kp = kps.transpose([1, 0, 2])
    hues = np.linspace(0, 360, num=nb_keypoints, endpoint=False, dtype=np.float32)
    colors = np.stack(
        [
            hues,
            np.ones(nb_keypoints, np.float32),
            np.ones(nb_keypoints, np.float32),
        ],
        axis=-1,
    )
    colors = np.squeeze(cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR))
    colors = (colors * 255).astype(np.uint8)
    for color, guesses in zip(colors, kps_by_kp):
        for kp in guesses:
            cv2.circle(image, tuple(kp[::-1]), 3, tuple(map(int, color)), -1)

    if err_rot is not None:
        cv2.putText(
            image,
            f"Rot Err: {np.degrees(err_rot):.2f}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
    if err_pos is not None:
        cv2.putText(
            image,
            f"Pos Err: {err_pos:.2f}",
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
