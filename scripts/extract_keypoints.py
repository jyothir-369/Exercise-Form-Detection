# scripts/extract_keypoints.py
import argparse
import numpy as np
import cv2
from pathlib import Path

# Fix imports by adding project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_detection.mediapipe_detector import MediaPipePoseDetector
from src.pose_detection.utils import extract_2d_keypoints


def infer_exercise_type(video_path):
    p = str(video_path).lower()
    if "bicep" in p or "curl" in p:
        return "bicep_curl"
    if "lateral" in p or "raise" in p:
        return "lateral_raise"
    if "posture" in p or "shoulder" in p or "stand" in p:
        return "posture"
    return "unknown"


def extract_keypoints(video_path, output_path=None):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    detector = MediaPipePoseDetector()
    exercise_type = infer_exercise_type(video_path)
    file_name = video_path.stem

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "keypoints" / exercise_type / f"{file_name}.npy"
    else:
        output_path = Path(output_path)

    print(f"Extracting keypoints...")
    print(f"Video: {video_path.name}")
    print(f"Exercise: {exercise_type}")
    print(f"Saving to: {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open video")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = 0
    keypoints_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 100 == 0 or frame_id <= 10:
            print(f"   Processing frame {frame_id}...")

        kp_norm, _ = detector.detect_keypoints(frame)

        if kp_norm is None:
            # Save zero-filled keypoints instead of None
            kp_2d = [(0, 0, 0.0)] * 33
        else:
            kp_2d = extract_2d_keypoints(kp_norm, frame_width, frame_height)

        keypoints_list.append(kp_2d)

    cap.release()

    # Convert to proper 3D numpy array
    keypoints_array = np.array(keypoints_list, dtype=np.float32)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, keypoints_array)

    print(f"SUCCESS! Saved {len(keypoints_list)} frames")
    print(f"File: {output_path}")
    print(f"Shape: {keypoints_array.shape} → (frames, 33, 3)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keypoints from video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=False, help="Custom output path")

    args = parser.parse_args()
    extract_keypoints(args.video, args.output)