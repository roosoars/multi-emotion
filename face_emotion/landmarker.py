"""Wrapper helpers around MediaPipe FaceLandmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from .constants import DEFAULT_LANDMARKER_MODEL

DetectionCallback = Callable[[object, int], None]


@dataclass
class LiveResultStore:
    """Thread-safe storage for the latest async result."""

    _result: object | None = None
    _timestamp_ms: int = -1

    def __post_init__(self) -> None:
        self._lock = Lock()

    def set(self, result: object, timestamp_ms: int) -> None:
        with self._lock:
            self._result = result
            self._timestamp_ms = timestamp_ms

    def get(self) -> tuple[object | None, int]:
        with self._lock:
            return self._result, self._timestamp_ms


def resolve_landmarker_model(path: str | Path | None = None) -> Path:
    model_path = Path(path) if path is not None else DEFAULT_LANDMARKER_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {model_path}. "
            "Make sure assets/face_landmarker.task is present."
        )
    return model_path


def bgr_frame_to_mp_image(frame: np.ndarray) -> mp.Image:
    """Convert an OpenCV BGR frame to a MediaPipe SRGB image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def create_face_landmarker(
    model_path: str | Path | None = None,
    running_mode: RunningMode = RunningMode.IMAGE,
    *,
    output_face_blendshapes: bool = True,
    result_callback: Callable[[object, mp.Image, int], None] | None = None,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> FaceLandmarker:
    """Create a FaceLandmarker for image, video or live-stream modes."""
    resolved = resolve_landmarker_model(model_path)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolved)),
        running_mode=running_mode,
        num_faces=1,
        output_face_blendshapes=output_face_blendshapes,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        result_callback=result_callback,
    )
    return FaceLandmarker.create_from_options(options)


class LiveStreamLandmarker:
    """Async live-stream wrapper around FaceLandmarker.detect_async."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        output_face_blendshapes: bool = True,
    ) -> None:
        self.store = LiveResultStore()

        def _on_result(result: object, _output_image: mp.Image, timestamp_ms: int) -> None:
            self.store.set(result, timestamp_ms)

        self._landmarker = create_face_landmarker(
            model_path=model_path,
            running_mode=RunningMode.LIVE_STREAM,
            output_face_blendshapes=output_face_blendshapes,
            result_callback=_on_result,
        )

    def submit(self, frame: np.ndarray, timestamp_ms: int) -> None:
        self._landmarker.detect_async(bgr_frame_to_mp_image(frame), timestamp_ms)

    def latest(self) -> tuple[object | None, int]:
        return self.store.get()

    def close(self) -> None:
        self._landmarker.close()
