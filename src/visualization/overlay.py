import cv2
import numpy as np

class OverlayRenderer:
    """
    Draws pose skeleton, keypoints, angles, and feedback on frames.
    """

    def __init__(self):
        # Define simple skeleton connections (MediaPipe Pose)
        self.connections = [
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 24), (11, 23), (12, 24)
        ]

    def draw_keypoints(self, frame, keypoints_2d):
        """
        Draw 2D keypoints on frame.
        """
        if keypoints_2d is None:
            return frame

        for (x, y, v) in keypoints_2d:
            if v > 0.5:  # visibility threshold
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        return frame

    def draw_skeleton(self, frame, keypoints_2d):
        """
        Draw skeleton connections.
        """
        if keypoints_2d is None:
            return frame

        for a, b in self.connections:
            x1, y1, v1 = keypoints_2d[a]
            x2, y2, v2 = keypoints_2d[b]

            if v1 > 0.5 and v2 > 0.5:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 150, 255), 2)

        return frame

    def draw_feedback(self, frame, feedback):
        """
        Draw feedback text on frame.
        feedback: dict from rule_engine
        """
        if not feedback:
            return frame

        y0 = 40
        dy = 30

        cv2.putText(frame, f"Exercise: {feedback.get('exercise', '')}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        y = y0 + dy

        # Handle list or string status
        statuses = feedback.get("status", "")
        if isinstance(statuses, str):
            statuses = [statuses]

        for status in statuses:
            cv2.putText(frame, status, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if "Correct" in status else (0, 0, 255), 
                        2)
            y += dy

        # Display angle if present
        if "angle" in feedback:
            cv2.putText(frame, f"Angle: {feedback['angle']:.1f}",
                        (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)

        return frame

    def render(self, frame, keypoints_2d, feedback):
        """
        Main rendering pipeline.
        """
        frame = self.draw_keypoints(frame, keypoints_2d)
        frame = self.draw_skeleton(frame, keypoints_2d)
        frame = self.draw_feedback(frame, feedback)
        return frame
