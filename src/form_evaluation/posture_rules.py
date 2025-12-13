# src/form_evaluation/posture_rules.py
from src.pose_detection.utils import calculate_angle


class PostureRules:
    def evaluate(self, keypoints):
        # MediaPipe landmark indices (x, y, visibility)
        left_shoulder = keypoints[11]   # Left shoulder
        right_shoulder = keypoints[12]  # Right shoulder
        left_hip = keypoints[23]        # Left hip
        right_hip = keypoints[24]       # Right hip

        # Calculate midpoints for better symmetry
        mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) // 2
        mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) // 2
        mid_hip_x = (left_hip[0] + right_hip[0]) // 2
        mid_hip_y = (left_hip[1] + right_hip[1]) // 2

        mid_shoulder = (mid_shoulder_x, mid_shoulder_y)
        mid_hip = (mid_hip_x, mid_hip_y)
        vertical_ref = (mid_hip_x, mid_hip_y + 200)  # Point straight down from hips

        # Calculate back alignment angle (180° = perfectly upright)
        back_angle = calculate_angle(mid_shoulder, mid_hip, vertical_ref)

        if 170 <= back_angle <= 190:
            return {
                "status": "correct",
                "message": "Excellent posture! Back is straight",
                "angle": round(back_angle, 1)
            }
        elif back_angle < 170:
            return {
                "status": "incorrect",
                "message": "Lean back slightly",
                "angle": round(back_angle, 1)
            }
        else:
            return {
                "status": "incorrect",
                "message": "Don't lean forward",
                "angle": round(back_angle, 1)
            }