# src/pose_detection/utils.py
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point B between points A-B-C.
    Accepts: (x,y), (x,y,v), or np.array — automatically handles 2D or 3D points.
    """
    # Extract only x,y — ignore visibility/z if present
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    # Avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))


def extract_2d_keypoints(keypoints, frame_width, frame_height):
    """
    Converts normalized MediaPipe landmarks (x,y,z,visibility) → pixel (x,y,visibility)
    """
    if keypoints is None:
        return None

    keypoints_2d = []
    for lm in keypoints:
        x = int(lm[0] * frame_width)
        y = int(lm[1] * frame_height)
        visibility = lm[3] if len(lm) > 3 else 1.0
        keypoints_2d.append((x, y, visibility))

    return keypoints_2d


def smooth_keypoints(buffer, window_size=5):
    """
    Simple moving average smoothing for keypoint history
    """
    if len(buffer) == 0:
        return None
    if len(buffer) < window_size:
        window_size = len(buffer)

    recent = buffer[-window_size:]
    arr = np.array(recent)
    smoothed = np.mean(arr, axis=0)
    return smoothed.tolist()