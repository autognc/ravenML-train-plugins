from scipy.spatial.transform import Rotation
import numpy as np
import cv2


def calculate_pose_vectors(ref_points, keypoints, focal_length, imdims,
                           rescale=1.0):
    """
    Calculates pose vectors using CV2's solvePNP.
    """
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    imdims = np.array(imdims)
    fy, fx = imdims // 2 * rescale
    focal_length *= rescale
    cam_matrix = np.array([
        [focal_length, 0, fx],
        [0, focal_length, fy],
        [0, 0, 1]
    ], dtype=np.float32)
    keypoints = keypoints.copy()
    keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
    ret, r_vec, t_vec = cv2.solvePnP(
        ref_points, keypoints,
        cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    assert ret, 'Pose solve failed.'
    return r_vec, t_vec, cam_matrix, dist_coeffs


def to_rotation(r):
    if isinstance(r, Rotation):
        return r
    r = np.array(r).squeeze()
    if len(r) == 3:
        return Rotation.from_rotvec(r)
    if len(r) == 4:
        w, x, y, z = r
        return Rotation.from_quat([x, y, z, w])
    raise ValueError()


def geodesic_error(rot_pred, rot_truth):
    rot_pred, rot_truth = to_rotation(rot_pred), to_rotation(rot_truth)
    # fix weird 180-degree flip
    rot_pred = Rotation.from_euler('xyz', [np.pi, 0, 0]) * rot_pred
    return _quat_geodesic_error(rot_pred.as_quat(), rot_truth.as_quat())


def _quat_geodesic_error(q1, q2):
    dot = np.abs(np.sum(q1 * q2))
    return 2 * np.arccos(np.clip(dot, 0, 1))
