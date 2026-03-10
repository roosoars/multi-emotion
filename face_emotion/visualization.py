"""OpenCV rendering helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from .constants import LANDMARK_REGIONS


def landmarks_to_pixels(landmarks: Iterable[object], width: int, height: int) -> list[tuple[int, int]]:
    """Convert normalized landmarks to integer pixel coordinates."""
    pixels: list[tuple[int, int]] = []
    for landmark in landmarks:
        x = int(float(landmark.x) * width)
        y = int(float(landmark.y) * height)
        pixels.append((x, y))
    return pixels


def bounding_box(points: Sequence[tuple[int, int]], margin: int = 20) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def _draw_contour(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    indices: Sequence[int],
    *,
    close_shape: bool,
) -> None:
    for start, end in zip(indices[:-1], indices[1:]):
        cv2.line(frame, points[start], points[end], (60, 220, 140), 1, cv2.LINE_AA)
    if close_shape and len(indices) > 2:
        cv2.line(frame, points[indices[-1]], points[indices[0]], (60, 220, 140), 1, cv2.LINE_AA)


def draw_face_mesh(frame: np.ndarray, landmarks: Iterable[object], *, draw_points: bool = True) -> tuple[int, int, int, int]:
    """Render landmarks and region contours. Returns the face bounding box."""
    height, width = frame.shape[:2]
    points = landmarks_to_pixels(landmarks, width, height)

    if draw_points:
        for point in points:
            cv2.circle(frame, point, 1, (0, 0, 255), -1, cv2.LINE_AA)

    for indices, close_shape in LANDMARK_REGIONS.values():
        _draw_contour(frame, points, indices, close_shape=close_shape)

    x1, y1, x2, y2 = bounding_box(points)
    draw_bounding_box(frame, (x1, y1, x2, y2))
    return x1, y1, x2, y2


def draw_bounding_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 80, 255), 2, cv2.LINE_AA)


def draw_prediction(
    frame: np.ndarray,
    label: str,
    confidence: float,
    *,
    location: tuple[int, int] = (20, 35),
    status: str | None = None,
) -> None:
    """Render prediction label and confidence overlay."""
    color = (0, 180, 0) if label != "Indefinida" else (0, 140, 255)
    cv2.putText(
        frame,
        f"Emocao: {label}",
        location,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confianca: {confidence * 100:.1f}%",
        (location[0], location[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            frame,
            status,
            (location[0], location[1] + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
