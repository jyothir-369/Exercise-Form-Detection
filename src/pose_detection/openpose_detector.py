# OPTIONAL — ONLY IF YOU HAVE OPENPOSE INSTALLED
# Install instructions: https://github.com/CMU-Perceptual-Computing-Lab/openpose

import cv2
import numpy as np
import sys
import os

class OpenPoseDetector:
    def __init__(self, model_folder="models/"):
        try:
            from openpose import pyopenpose as op
        except ImportError as e:
            raise Exception("OpenPose not installed. Install pyopenpose first.")

        params = dict()
        params["model_folder"] = model_folder

        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()

        self.datum = op.Datum()

    def detect_keypoints(self, frame):
        """Runs OpenPose on a frame and returns keypoints."""
        self.datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop([self.datum])

        if self.datum.poseKeypoints is None:
            return None, None
        
        keypoints = self.datum.poseKeypoints[0]  # first detected person
        return keypoints, self.datum

    def draw_pose(self, frame):
        """Draw OpenPose skeleton on frame."""
        return self.datum.cvOutputData
