import cv2
import argparse
import os
import numpy as np

from pose_detection.mediapipe_detector import MediaPipePoseDetector
from form_evaluation.rule_engine import RuleEngine
from visualization.overlay import OverlayRenderer
from visualization.smoothing import KeypointSmoother
from pose_detection.utils import extract_2d_keypoints

# Optional MLflow
try:
    from mlflow_tracking.mlflow_logger import MLFlowLogger
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False


# -------------------------------------------------------------------
# AUTO VIDEO LOADER (NEW)
# -------------------------------------------------------------------

VIDEO_ROOT = "data/raw/youtube_videos"

def get_all_videos():
    """Scan all subfolders under youtube_videos and return mp4 file paths."""
    for root, dirs, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.lower().endswith(".mp4"):
                yield os.path.join(root, f)

def infer_exercise_type(video_path):
    """Detect exercise type based on folder name."""
    p = video_path.replace("\\", "/").lower()

    if "bicep" in p:
        return "bicep_curl"
    if "lateral" in p:
        return "lateral_raise"
    if "posture" in p:
        return "posture"

    return None


# -------------------------------------------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------------------------------------------

def run_pipeline(video_path, exercise_type, output_path="output/overlays/output.mp4", use_mlflow=False):

    # Initialize modules
    detector = MediaPipePoseDetector()
    rule_engine = RuleEngine()
    overlay = OverlayRenderer()
    smoother = KeypointSmoother(window_size=5, method="moving_average")

    # MLflow logging
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow_logger = MLFlowLogger()
        mlflow_logger.log_params({"exercise_type": exercise_type})
    else:
        mlflow_logger = None

    # Read input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file: {video_path}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    print(f"🎯 Running evaluation for {exercise_type} on: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose detection
        keypoints_norm, landmarks = detector.detect_keypoints(frame)

        if keypoints_norm is None:
            writer.write(frame)
            continue

        # Convert normalized → pixel
        keypoints_2d = extract_2d_keypoints(keypoints_norm, frame_width, frame_height)

        # Smooth keypoints
        keypoints_2d_array = np.array([kp[:2] for kp in keypoints_2d])
        smoothed = smoother.smooth(keypoints_2d_array)

        for i in range(len(keypoints_2d)):
            keypoints_2d[i] = (int(smoothed[i][0]), int(smoothed[i][1]), keypoints_2d[i][2])

        # Rule engine evaluation
        feedback = rule_engine.evaluate(exercise_type, keypoints_2d)

        # MLflow logging
        if mlflow_logger and "angle" in feedback:
            mlflow_logger.log_metrics({"angle": feedback["angle"]})

        # Visualization
        annotated = overlay.render(frame.copy(), keypoints_2d, feedback)

        writer.write(annotated)

        cv2.imshow("Exercise Form Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"✅ Output saved to: {output_path}")

    if mlflow_logger:
        mlflow_logger.end()
        print("📊 MLflow logging saved.")


# -------------------------------------------------------------------
# PROGRAM ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise Form Detection Pipeline")

    parser.add_argument(
        "--video",
        type=str,
        required=False,
        help="Path to input video file. If not provided, auto-loads all videos from youtube_videos."
    )

    parser.add_argument(
        "--exercise",
        type=str,
        required=False,
        choices=["bicep_curl", "lateral_raise", "posture"],
        help="If not provided, detected automatically based on folder name."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated video. Auto-names if not provided."
    )

    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")

    args = parser.parse_args()


    # Case 1 — Manual video path provided
    if args.video:
        exercise = args.exercise or infer_exercise_type(args.video)
        out = args.output or f"output/overlays/{os.path.basename(args.video)}"
        run_pipeline(args.video, exercise, out, use_mlflow=args.mlflow)

    # Case 2 — No video supplied: automatically load ALL dataset videos
    else:
        print("🔍 No --video provided. Auto-loading videos from youtube_videos/...")

        for video in get_all_videos():
            exercise = infer_exercise_type(video)
            if exercise is None:
                print(f"⚠ Skipping file (cannot determine exercise): {video}")
                continue

            output_file = f"output/overlays/{exercise}_{os.path.basename(video)}"
            run_pipeline(video, exercise, output_file, use_mlflow=args.mlflow)
