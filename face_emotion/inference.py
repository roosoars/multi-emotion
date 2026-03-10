"""Inference helpers for real-time face emotion recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from .constants import CONFIDENCE_THRESHOLD, DEFAULT_EMOTION_MODEL
from .features import build_feature_vector, compute_geometric_features, encode_blendshapes, landmarks_to_array, normalize_landmarks


@dataclass
class EmotionPrediction:
    """Final prediction for a frame."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    raw_label: str
    threshold_met: bool


class EmotionClassifier:
    """Load the trained artifact and produce probability-based predictions."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        resolved = Path(artifact_path) if artifact_path is not None else DEFAULT_EMOTION_MODEL
        if not resolved.exists():
            raise FileNotFoundError(
                f"Emotion model not found at {resolved}. Run train.py before starting inference."
            )
        artifact = joblib.load(resolved)
        self.estimator = artifact["model"]
        self.labels = artifact["labels"]
        self.feature_set = artifact["feature_set"]
        self.blendshape_names = artifact["blendshape_names"]
        self.threshold = float(artifact.get("confidence_threshold", CONFIDENCE_THRESHOLD))
        self.metadata = artifact

    def vector_from_detection(self, face_landmarks: Sequence[object], blendshapes: Sequence[object]) -> np.ndarray:
        landmarks = landmarks_to_array(face_landmarks)
        normalized = normalize_landmarks(landmarks)
        geometry = compute_geometric_features(normalized)
        blendshape_vector, _ = encode_blendshapes(blendshapes, self.blendshape_names)
        return build_feature_vector(normalized, geometry, blendshape_vector, self.feature_set)

    def predict_probabilities(self, face_landmarks: Sequence[object], blendshapes: Sequence[object]) -> np.ndarray:
        vector = self.vector_from_detection(face_landmarks, blendshapes)
        return self.estimator.predict_proba(vector.reshape(1, -1))[0]

    def predict_from_detection(
        self,
        face_landmarks: Sequence[object],
        blendshapes: Sequence[object],
        *,
        probabilities_override: Sequence[float] | None = None,
    ) -> EmotionPrediction:
        if probabilities_override is None:
            probabilities = self.predict_probabilities(face_landmarks, blendshapes)
        else:
            probabilities = np.array(list(probabilities_override), dtype=np.float32)

        probabilities_dict = {label: float(score) for label, score in zip(self.labels, probabilities)}
        best_index = int(np.argmax(probabilities))
        raw_label = self.labels[best_index]
        confidence = float(probabilities[best_index])
        label = raw_label if confidence >= self.threshold else "Indefinida"
        return EmotionPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities_dict,
            raw_label=raw_label,
            threshold_met=confidence >= self.threshold,
        )
