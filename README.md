# Multimodal Emotion Recognition

A real-time facial landmark detection project built with MediaPipe Face Mesh and OpenCV.

This repository currently implements the foundational computer vision layer for a broader emotion recognition system. At this stage, the application detects a single face in real time, extracts 478 facial landmarks, and renders facial contours, iris contours, and a bounding box over the webcam stream.

## Requirements

- Python 3.8+
- Webcam
- macOS, Linux, or Windows

## Installation

For Linux/macOS

```bash
git clone https://github.com/roosoars/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Press `ESC` to close the application.

## License

This project is licensed under the MIT License.
