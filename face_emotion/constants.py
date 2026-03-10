"""Shared constants for training and inference."""

from __future__ import annotations

from pathlib import Path

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
]

CONFIDENCE_THRESHOLD = 0.60
SMOOTHING_WINDOW = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANDMARKER_MODEL = PROJECT_ROOT / "assets" / "face_landmarker.task"
DEFAULT_EMOTION_MODEL = PROJECT_ROOT / "artifacts" / "emotion_model.joblib"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "artifacts" / "cache" / "features.joblib"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LANDMARK_REGIONS = {
    "left_eye": (LEFT_EYE, True),
    "right_eye": (RIGHT_EYE, True),
    "left_eyebrow": (LEFT_EYEBROW, False),
    "right_eyebrow": (RIGHT_EYEBROW, False),
    "lips_outer": (LIPS_OUTER, True),
    "lips_inner": (LIPS_INNER, True),
    "face_oval": (FACE_OVAL, True),
    "nose": (NOSE, False),
    "left_iris": (LEFT_IRIS, True),
    "right_iris": (RIGHT_IRIS, True),
}
