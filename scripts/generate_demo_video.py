# scripts/generate_demo_video.py
import argparse
import cv2
import numpy as np
import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# FIX: Add project root so src/ imports work perfectly
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

# Now ALL imports work flawlessly
from src.pose_detection.mediapipe_detector import MediaPipePoseDetector
from src.pose_detection.utils import extract_2d_keypoints
from src.form_evaluation.rule_engine import RuleEngine
from src.visualization.smoothing import KeypointSmoother
from src.visualization.overlay import OverlayRenderer  # You'll create this next!


# ──────────────────────────────────────────────────────────────
# AUTO DETECT EXERCISE TYPE
# ──────────────────────────────────────────────────────────────
def infer_exercise_type(video_path):
    p = str(video_path).lower()
    if "bicep" in p or "curl" in p:
        return "bicep_curl"
    if "lateral" in p or "raise" in p:
        return "lateral_raise"
    if "posture" in p or "shoulder" in p or "stand" in p:
        return "posture"
    return None


# ──────────────────────────────────────────────────────────────
# MAIN DEMO GENERATION — CINEMATIC QUALITY
# ──────────────────────────────────────────────────────────────
def generate_demo(video_path, exercise_type=None, output_path=None):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    exercise_type = exercise_type or infer_exercise_type(video_path)
    if not exercise_type:
        raise ValueError("Could not detect exercise type. Use --exercise flag.")

    # Auto output path
    if output_path is None:
        safe_name = video_path.stem
        output_path = PROJECT_ROOT / "output" / "overlays" / exercise_type / f"{safe_name}_DEMO.mp4"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Cannot open video")
        return

    # Initialize components
    detector = MediaPipePoseDetector()
    renderer = OverlayRenderer()
    smoother = KeypointSmoother(window_size=7)
    rule_engine = RuleEngine()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Generating DEMO VIDEO...")
    print(f"Exercise: {exercise_type.replace('_', ' ').title()}")
    print(f"Input:  {video_path.name}")
    print(f"Output: {output_path}")
    print(f"Resolution: {width}×{height} @ {fps:.1f} FPS\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Rendering frame {frame_count}...")

        # Detect + convert keypoints
        kp_norm, _ = detector.detect_keypoints(frame)
        if kp_norm is None:
            writer.write(frame)
            continue

        kp_2d = extract_2d_keypoints(kp_norm, width, height)

        # Smooth for silky motion
        coords = np.array([p[:2] for p in kp_2d])
        smoothed = smoother.smooth(coords)
        for i in range(len(kp_2d)):
            kp_2d[i] = (int(smoothed[i][0]), int(smoothed[i][1]), kp_2d[i][2])

        # Get real-time feedback
        feedback = rule_engine.evaluate(exercise_type, kp_2d)

        # Render BEAUTIFUL overlay
        annotated_frame = renderer.render(frame.copy(), kp_2d, feedback)

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    print(f"\nDEMO VIDEO COMPLETE!")
    print(f"Saved: {output_path}")
    print(f"Ready for submission!")


# ──────────────────────────────────────────────────────────────
# RUN IT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate beautiful exercise form demo video")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--exercise", type=str, choices=["bicep_curl", "lateral_raise", "posture"],
                        help="Exercise type (optional, auto-detected)")
    parser.add_argument("--output", type=str, help="Output path (optional)")

    args = parser.parse_args()
    generate_demo(args.video, args.exercise, args.output)