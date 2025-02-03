# Remote Controlled Robot Navigation

An end-to-end deep learning project for autonomous RC robot navigation using behavioral cloning. The system learns to map raw camera inputs directly to steering and throttle controls by mimicking human driving behavior.

## Architecture Overview

### Neural Network (RCCarNet)
- CNN architecture inspired by NVIDIA's self-driving car model
- 5 convolutional layers with batch normalization
- 4 fully connected layers with dropout
- Dual output: steering (1000-2000) and throttle (1000-2000)

### Data Pipeline
- Input: 66x200 RGB images from car's perspective
- Output: Normalized control values [-1,1]
- Real-time inference at 30+ FPS

### Key Features
- End-to-end behavioral cloning
- Real-time trajectory visualization
- Support for both image and video inference
- Batch normalization for training stability
- Dropout layers for regularization

## Model Architecture Details

### Convolutional Layers (CNN)
1. Conv2d(3, 24, 5, stride=2) → BatchNorm2d
2. Conv2d(24, 36, 5, stride=2) → BatchNorm2d
3. Conv2d(36, 48, 5, stride=2) → BatchNorm2d
4. Conv2d(48, 64, 3) → BatchNorm2d
5. Conv2d(64, 64, 3) → BatchNorm2d

### Fully Connected Layers (FC)
1. Linear(1152, 100) → ReLU → Dropout(0.5)
2. Linear(100, 50) → ReLU → Dropout(0.5)
3. Linear(50, 10) → ReLU
4. Linear(10, 2) → Tanh

## Usage
1. Train model with collected driving data
2. Run inference on images: `python rc_robot_inference_v2.py`
3. Process videos: `python rc_robot_video_inference.py`

## Dependencies
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL

## Acknowledgments
- NVIDIA's End-to-End Deep Learning for Self-Driving Cars paper