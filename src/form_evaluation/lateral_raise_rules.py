from ..pose_detection.utils import calculate_angle
import numpy as np

class LateralRaiseRules:
    """
    Rule-based checks for lateral raise.
    """

    def __init__(self, min_angle=70, max_angle=110):
        """
        min_angle -> lower bound for shoulder abduction
        max_angle -> upper bound (to avoid raising too high)
        """
        self.min_angle = min_angle
        self.max_angle = max_angle

    def evaluate(self, keypoints_2d):
        LEFT_SHOULDER = 11
        LEFT_ELBOW = 13
        LEFT_WRIST = 15

        try:
            shoulder = keypoints_2d[LEFT_SHOULDER][:2]
            elbow = keypoints_2d[LEFT_ELBOW][:2]
            wrist = keypoints_2d[LEFT_WRIST][:2]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Additional rule —
            # Shoulder & wrist should be horizontally aligned
            vertical_diff = abs(wrist[1] - shoulder[1])

            if angle < self.min_angle:
                status = "Raise arm higher."
            elif angle > self.max_angle:
                status = "Arm raised too high; keep at shoulder level."
            elif vertical_diff > 40:
                status = "Keep wrist at shoulder height (avoid dropping wrist)."
            else:
                status = "Correct form"

            return {
                "exercise": "lateral_raise",
                "angle": angle,
                "status": status
            }

        except:
            return {
                "exercise": "lateral_raise",
                "status": "keypoints_missing"
            }
