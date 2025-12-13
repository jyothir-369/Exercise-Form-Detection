import numpy as np
from collections import deque
from scipy.signal import savgol_filter

class KeypointSmoother:
    """
    Smooths 2D keypoint time series.
    """

    def __init__(self, window_size=5, method="moving_average"):
        self.window_size = window_size
        self.method = method
        self.buffer = deque(maxlen=window_size)

    def smooth(self, keypoints):
        """
        keypoints: np.array of shape (33, 3/4)
        Returns smoothed keypoints using chosen method.
        """
        self.buffer.append(keypoints)

        if len(self.buffer) < self.window_size:
            return keypoints

        data = np.array(self.buffer)

        if self.method == "moving_average":
            return np.mean(data, axis=0)

        elif self.method == "savgol":
            # Apply Savitzky-Golay filter across time axis
            return savgol_filter(data, window_length=self.window_size,
                                 polyorder=2, axis=0)

        else:
            return keypoints
