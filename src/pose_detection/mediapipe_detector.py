import cv2
import mediapipe as mp
import numpy as np

class MediaPipePoseDetector:
    def __init__(self,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def detect_keypoints(self, frame):
        """Runs pose detection on a single frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks.landmark
        keypoints = []

        for lm in landmarks:
            keypoints.append([lm.x, lm.y, lm.z, lm.visibility])

        return np.array(keypoints), results.pose_landmarks

    def draw_pose(self, frame, pose_landmarks):
        """Draw keypoints and skeleton on a frame."""
        if pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        return frame
