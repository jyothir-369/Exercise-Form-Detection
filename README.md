Markdown# AI Intern — Exercise Form Detection

This project implements a pose-estimation-based pipeline to detect exercise form correctness using **MediaPipe** (default), **OpenPose**, or any human-pose model. It includes:

- Pose keypoint extraction
- Angle computation
- Rule-based posture evaluation
- Real-time feedback overlay on video
- Optional MLflow integration

![MediaPipe Pose - 33 Full Body Landmarks (Official Diagram)](https://camo.githubusercontent.com/034c02b2e6aae3873f5a4dba10fc7a200ad5b161396f25709f07109df8ff1067/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67)

---

## 📌 Project Structure

AI-Intern-Exercise-Form-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── pose_detection/
│   │   ├── mediapipe_detector.py
│   │   ├── openpose_detector.py
│   │   └── utils.py
│   │
│   ├── form_evaluation/
│   │   ├── bicep_curl_rules.py
│   │   ├── lateral_raise_rules.py
│   │   ├── posture_rules.py
│   │   └── rule_engine.py
│   │
│   ├── visualization/
│   │   ├── overlay.py
│   │   └── smoothing.py
│   │
│   ├── mlflow_tracking/
│   │   └── mlflow_logger.py
│   │
│   └── main.py
│
├── scripts/
│   ├── extract_keypoints.py
│   ├── evaluate_video.py
│   └── generate_demo_video.py
│
├── notebooks/
│   ├── exploratory_pose.ipynb
│   └── angle_calculation_tests.ipynb
│
├── data/
│   ├── raw/
│   │   ├── coco2017/
│   │   ├── mpii/
│   │   ├── fitness_dataset/
│   │   └── youtube_videos/
│   │
│   ├── processed/
│   └── keypoints/
│
├── output/
│   ├── results/
│   ├── overlays/
│   ├── logs/
│   └── mlflow/
│
└── docs/
    ├── Report.pdf
    └── posture_rules_explained.md

## 📌 Features

### ✔ Pose Estimation
Uses **MediaPipe** (default) or **OpenPose** to extract 33 (MediaPipe) or 18 (OpenPose) keypoints in real-time.

### ✔ Angle Computation
Joint angles (e.g., elbow flexion, shoulder abduction) are calculated using the vector dot product method.

![Joint Angle Calculation using Vector Dot Product (Shoulder/Elbow Example)](https://www.mdpi.com/sensors/sensors-24-02912/article_deploy/html/images/sensors-24-02912-g005-550.jpg)

### ✔ Form Evaluation Rules
Exercise-specific rule-based checks:
- **Bicep Curl**: Elbow flexion angle and arm alignment
- **Lateral Raise**: Shoulder abduction/elevation and wrist-shoulder symmetry
- **Posture Correction**: Spine straightness, shoulder symmetry, and forward head detection

#### Bicep Curl Detection Example
![Pose Skeleton Overlay on Bicep Curl Exercise](https://dl.acm.org/cms/attachment/html/10.1145/3591156.3591168/assets/html/images/image3.png)

#### Lateral Raise Detection Example
![Pose Skeleton Overlay on Lateral Raise Exercise](https://www.mdpi.com/applsci/applsci-10-00611/article_deploy/html/images/applsci-10-00611-g003.png)

#### Posture Correction Example
![Pose Estimation for Posture Analysis (Forward Head and Slouched Back Detection)](https://www.caringmedical.com/wp-content/uploads/2020/09/forward-head-posture-WEB.png)

### ✔ Keypoint Smoothing
Time-series smoothing using moving average or Savitzky-Golay filter to reduce jitter.

### ✔ Real-Time Feedback
Generates annotated videos with:
- Neon skeleton overlay
- Live angle display
- "Correct" (green) / "Incorrect" (red) labels and messages

![Real-Time Form Feedback with Skeleton, Angles, and Correct/Incorrect Labels](https://aicertswpcdn.blob.core.windows.net/newsportal/2025/11/pose-estimation-in-action.jpg)

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
Step 2: Run the Pipeline
Bashpython src/main.py --video data/raw/youtube_videos/your_exercise_video.mp4
Step 3: View Output
Annotated demo videos are saved in:
textoutput/overlays/

📁 Dataset
Primary testing on YouTube workout tutorials. Compatible with:

Self-recorded short clips
Kaggle fitness datasets
COCO / MPII pose datasets (for validation)


📄 Submission Includes

Full Python source code
Annotated demo videos with overlays
Detailed documentation and posture rule explanations


👤 Author
Jyothir Raghavalu Bhogi
Date: December 2025
Submitted for: Smartan Fitech Private Limited – Computer Vision & AI Internship Task
GitHub: https://github.com/jyothir-369/AI-Intern-Exercise-Form-Detection