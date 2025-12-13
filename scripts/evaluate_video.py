# scripts/evaluate_video.py
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# ULTIMATE IMPORT FIX — Works even with broken relative imports
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root AND src directory to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Now ALL imports work — no matter how messy your internal imports are
from src.pose_detection.mediapipe_detector import MediaPipePoseDetector
from src.pose_detection.utils import extract_2d_keypoints
from src.form_evaluation.rule_engine import RuleEngine
from src.visualization.smoothing import KeypointSmoother

import argparse
import cv2
import numpy as np


def infer_exercise_type(video_path: str) -> str:
    p = str(video_path).lower().replace("\\", "/")
    if any(x in p for x in ["bicep", "curl"]):
        return "bicep_curl"
    if any(x in p for x in ["lateral", "raise"]):
        return "lateral_raise"
    if any(x in p for x in ["posture", "stand", "spine", "back", "shoulder"]):
        return "posture"
    return None


def evaluate_video(video_path: str, exercise_type: str, output_txt: str):
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\nStarting form evaluation...")
    print(f"Video    : {video_path.name}")
    print(f"Exercise : {exercise_type.replace('_', ' ').title()}\n")

    detector = MediaPipePoseDetector()
    rule_engine = RuleEngine()
    smoother = KeypointSmoother()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Resolution: {width}×{height} | Frames: {total_frames} | FPS: {fps:.2f}\n")

    results = []
    frame_id = 0
    correct_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        kp_norm, _ = detector.detect_keypoints(frame)
        if kp_norm is None:
            results.append(f"Frame {frame_id}: No person detected")
            continue

        kp_2d = extract_2d_keypoints(kp_norm, width, height)

        # Smooth keypoints
        coords = np.array([kp[:2] for kp in kp_2d])
        smoothed = smoother.smooth(coords)
        for i in range(len(kp_2d)):
            kp_2d[i] = (int(smoothed[i][0]), int(smoothed[i][1]), kp_2d[i][2])

        # Evaluate form
        feedback = rule_engine.evaluate(exercise_type, kp_2d)
        status = feedback.get("status", "unknown").lower()

        if status in ("correct", "good"):
            correct_frames += 1

        if "angle" in feedback:
            line = f"Frame {frame_id}: angle={feedback['angle']:.1f}°, status={status}, msg='{feedback.get('message','')}'"
        else:
            line = f"Frame {frame_id}: status={status}, msg='{feedback.get('message','')}'"

        results.append(line)

        if frame_id % 50 == 0 or frame_id == total_frames:
            print(f"   Processed {frame_id}/{total_frames} frames...")

    cap.release()

    accuracy = (correct_frames / frame_id * 100) if frame_id > 0 else 0

    summary = f"""
=== EVALUATION SUMMARY ===
Video           : {video_path.name}
Exercise        : {exercise_type.replace('_', ' ').title()}
Total Frames    : {frame_id}
Correct Frames  : {correct_frames}
Accuracy        : {accuracy:.2f}%
================================
"""

    print(summary)

    out_path = Path(output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary.strip() + "\n\n" + "\n".join(results))

    print(f"Report saved: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Exercise Form Evaluator")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--exercise", type=str, choices=["bicep_curl", "lateral_raise", "posture"],
                        help="Exercise type (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="output/results/evaluation.txt",
                        help="Output report path")

    args = parser.parse_args()

    exercise = args.exercise or infer_exercise_type(args.video)
    if not exercise:
        print("Could not detect exercise type!")
        print("Please use --exercise bicep_curl | lateral_raise | posture")
        sys.exit(1)

    evaluate_video(args.video, exercise, args.output)