from ..pose_detection.utils import calculate_angle
class BicepCurlRules:
    """
    Rule-based evaluation for bicep curl form.
    """

    def __init__(self, elbow_min=40, elbow_max=160):
        """
        elbow_min -> angle at top of curl (arm contracted)
        elbow_max -> angle at bottom of curl (arm extended)
        """
        self.elbow_min = elbow_min
        self.elbow_max = elbow_max

    def evaluate(self, keypoints_2d):
        """
        keypoints_2d -> list of (x, y, visibility) for 33 MediaPipe keypoints
        Returns: dict { "status": "correct"/"incorrect", "angle": value }
        """

        # MediaPipe keypoint indices
        # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        RIGHT_SHOULDER = 12
        RIGHT_ELBOW = 14
        RIGHT_WRIST = 16

        try:
            shoulder = keypoints_2d[RIGHT_SHOULDER][:2]
            elbow = keypoints_2d[RIGHT_ELBOW][:2]
            wrist = keypoints_2d[RIGHT_WRIST][:2]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < self.elbow_min:
                status = "Incomplete curl — lift higher."
            elif angle > self.elbow_max:
                status = "Arm over-extended — avoid locking elbow."
            else:
                status = "Correct form"

            return {
                "exercise": "bicep_curl",
                "angle": angle,
                "status": status
            }

        except:
            return {
                "exercise": "bicep_curl",
                "status": "keypoints_missing"
            }
