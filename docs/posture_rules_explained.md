# Posture Rules — Explained

This document describes the rule-based posture checks used in the **AI-Intern Exercise Form Detection** project.  
It lists each rule, the reasoning behind it, the exact geometric logic (keypoints/angles), thresholds, and failure messages.

---

## Key conventions

- Keypoints follow MediaPipe indexing (0..32). See reference mapping in `notebooks/exploratory_pose.ipynb`.  
- `keypoints_2d[i] = (x, y, visibility)` — pixel coordinates and visibility confidence.  
- All angles are computed at the middle joint using the three-point formula:
  `angle = angle(a, b, c)` where `b` is the vertex.

---

## 1. Bicep Curl Rules

**Goal:** check elbow flexion/extension and avoid elbow lock / incomplete range.

**Keypoints used**
- `RIGHT_SHOULDER` (12)
- `RIGHT_ELBOW` (14)
- `RIGHT_WRIST` (16)

**Logic**
- Compute elbow angle: `angle = angle(shoulder, elbow, wrist)`.
- Expected range (typical):  
  - Fully contracted (top): `~30°–50°` (elbow angle small).  
  - Fully extended (bottom): `~150°–180°`.

**Thresholds (configurable)**
- `elbow_min = 40°` (if angle < 40 → "Incomplete curl — lift higher.")
- `elbow_max = 160°` (if angle > 160 → "Arm over-extended — avoid locking elbow.")
- `40° <= angle <= 160°` → "Correct form"

**Notes / rationale**
- People have natural variance in range of motion. Use `elbow_min` and `elbow_max` as tunable parameters.
- If visibility of any keypoint < 0.4, mark `keypoints_missing`.

---

## 2. Lateral Raise Rules

**Goal:** ensure arm is raised laterally to shoulder height, without shrugging or dropping the wrist.

**Keypoints used**
- `LEFT_SHOULDER` (11)
- `LEFT_ELBOW` (13)
- `LEFT_WRIST` (15)

**Logic**
- Compute arm angle about elbow: `angle = angle(shoulder, elbow, wrist)`. This measures arm abduction.
- Check vertical alignment: `vertical_diff = abs(wrist_y - shoulder_y)` to ensure wrist is near shoulder height.
- Typical expected angle around `80°–100°` for a proper lateral raise.

**Thresholds**
- `min_angle = 70°` → if angle < 70: "Raise arm higher."
- `max_angle = 110°` → if angle > 110: "Arm raised too high; keep at shoulder level."
- `vertical_diff_threshold = 40 px` → if vertical difference too large: "Keep wrist at shoulder height."

**Notes**
- Use pixel thresholds for `vertical_diff` — adjust according to camera distance and resolution.
- An alternative is to convert pixels to normalized coordinates and use a percentage-of-frame threshold.

---

## 3. Posture / Back Rules

**Goal:** detect spine curvature, shoulder or hip asymmetry and gross forward/backward lean.

**Keypoints used**
- `LEFT_SHOULDER` (11), `RIGHT_SHOULDER` (12)
- `LEFT_HIP` (23), `RIGHT_HIP` (24)

**Logic**
1. **Shoulder alignment**
   - `shoulder_diff = abs(left_shoulder_y - right_shoulder_y)`  
   - If `shoulder_diff > 25 px` → "Shoulders uneven — keep both level."

2. **Hip alignment**
   - `hip_diff = abs(left_hip_y - right_hip_y)`  
   - If `hip_diff > 25 px` → "Hips uneven — maintain stability."

3. **Back angle (approximate)**
   - Use vector from shoulder to hip (e.g., left_shoulder - left_hip).
   - Compute angle against vertical: `angle_back = angle_between(back_vector, vertical)`.
   - If `angle_back > 20°` → "Back bent — keep spine neutral."

**Notes**
- The back angle heuristic is simple and works well for frontal or slightly angled views. For more accurate spine curvature detection, use multi-view or 3D keypoints (e.g., Human3.6M or Fit3D).
- Thresholds are conservative; you can relax/tighten based on dataset and camera placement.

---

## 4. Multi-Person Handling (recommended approach)

**Goal:** correctly handle frames containing multiple persons.

**Approach**
1. **Person detection + association**
   - If using MediaPipe Single-Pose: falls back to first detection (not ideal). Prefer multi-person capable model (OpenPose or BlazePose GHUM Multi-pose).
2. **Per-person pipeline**
   - For each detected person, extract keypoints and compute rules independently.
   - Keep local buffers for smoothing per person (by id).
3. **ID tracking**
   - Use a simple tracker (centroid, IOU on bounding boxes, or SORT/DeepSORT) to maintain consistent person IDs across frames.
4. **Display**
   - Render a small label `Person #k` and per-person feedback near that person’s bounding box.

**Edge cases**
- Occlusion / overlapping persons → decrease in keypoint visibility; set `keypoints_missing`.
- Multiple people with similar poses → track with spatial tracking and temporal smoothing to avoid jitter.

---

## 5. Smoothing & Temporal Rules

**Smoothing**
- Use moving average over a short window (default 5 frames) or Savitzky–Golay for better derivative behavior.
- Smooth pixel coordinates independently across the time axis before computing angles.

**Temporal checks**
- For repetitions (e.g., bicep curl), compute a small state machine:
  - `down` → `up` → `down` transitions using elbow angle thresholds to count reps.
  - Use hysteresis on thresholds to avoid noise-triggered transitions.

---

## 6. Implementation & Tuning Tips

- **Camera placement**: frontal or 3/4 view works best. Avoid strong perspective distortion.
- **Calibration**: convert pixels → meters if you have camera calibration for distance-based thresholds.
- **Visibility**: if any relevant keypoint visibility < 0.4, do not compute angle — return `keypoints_missing`.
- **Threshold selection**: record a small validation set (self-recorded) and choose thresholds that maximize correct/incorrect detection.
- **Logging**: store frame-wise angles and statuses for later analysis (helpful when integrating MLflow).

---

## 7. Example messages returned by the rule engine

- `"Correct form"`  
- `"Incomplete curl — lift higher."`  
- `"Arm over-extended — avoid locking elbow."`  
- `"Raise arm higher."`  
- `"Arm raised too high; keep at shoulder level."`  
- `"Keep wrist at shoulder height (avoid dropping wrist)."`  
- `"Shoulders uneven — keep both level."`  
- `"Back bent — keep spine neutral."`  
- `"keypoints_missing"`

---

## 8. Future improvements

- Convert 2D rules to 3D using multi-view / MoCap datasets (Human3.6M, Fit3D).  
- Replace heuristic thresholds with a small classifier trained on labeled correct/incorrect reps.  
- Add personalized thresholds: auto-calibrate based on user's limb length and camera distance.

