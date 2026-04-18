# 401_selfdefined_project
Project Overview
This project implements a lightweight, camera-based LEGO brick recognition system using YOLOv8, a state-of-the-art one-stage object detection model. The system is designed to simulate industrial automated inspection, identifying 4 common LEGO brick classes (2x1, rec, cir, squ) in real time from live camera feed.
The goal is to solve the limitations of traditional rule-based machine vision (e.g., poor robustness to lighting, angle changes, and partial occlusion) with a deep learning pipeline that is fast, accurate, and easy to deploy on consumer hardware.

Project Objectives
Build an end-to-end real-time LEGO brick detection system using deep learning
Achieve high classification accuracy while maintaining 30+ FPS on a standard laptop
Validate the system’s robustness under variable lighting, angles, and partial occlusion
Deliver a reproducible, well-documented pipeline suitable for a professional technical portfolio

Tech Stack
Programming Language: Python 3.10+
Deep Learning Framework: Ultralytics YOLOv8
Computer Vision: OpenCV
Hardware: USB webcam (or built-in laptop camera), Intel Core i5/i7 laptop

Results & Key Findings
The final system achieves strong performance on both accuracy and speed, validated in controlled and real-world conditions:
Metric	Value
Overall Classification Accuracy	96.42%
Average Inference Speed	32.5 FPS
Valid Detection Success Rate	98.93%
Accuracy Under Variable Lighting	87.14%
Accuracy Under ±10° Camera Tilt	89.29%

Key takeaways from the project:
YOLOv8 outperforms traditional rule-based machine vision in robustness to environmental variations
Adding a region-of-interest (ROI) crop significantly reduces background noise and improves FPS
The lightweight YOLOv8n model is sufficient for real-time deployment on consumer laptops, without requiring a GPU
