"""Training and validation pipeline for face emotion recognition."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEFAULT_CACHE_PATH,
    DEFAULT_LANDMARKER_MODEL,
    DEFAULT_REPORT_DIR,
    SMOOTHING_WINDOW,
)
from .features import (
    GEOMETRIC_FEATURE_NAMES,
    build_feature_vector,
    compute_geometric_features,
    encode_blendshapes,
    labels_to_index,
    landmarks_to_array,
    normalize_landmarks,
)
from .landmarker import create_face_landmarker


@dataclass
class FeatureRecord:
    split: str
    label: str
    image_path: str
    detected: bool
    dense_features: np.ndarray | None
    geometric_features: np.ndarray | None
    blendshape_features: np.ndarray | None


@dataclass
class CandidateSpec:
    name: str
    feature_set: str
    estimator_factory: Callable[[], object]


def _image_files(dataset_root: Path) -> list[tuple[str, str, Path]]:
    files: list[tuple[str, str, Path]] = []
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        for label in CLASS_NAMES:
            label_dir = split_dir / label
            for path in sorted(label_dir.glob("*")):
                if path.is_file() and path.suffix.lower() in allowed_suffixes and not path.name.startswith("."):
                    files.append((split, label, path))
    return files


def _extract_features_for_image(
    image_path: Path,
    detector: object,
    blendshape_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    image = mp.Image.create_from_file(str(image_path))
    result = detector.detect(image)
    if not result.face_landmarks:
        return None

    dense = normalize_landmarks(landmarks_to_array(result.face_landmarks[0]))
    geometry = compute_geometric_features(dense)
    blendshape_vector, resolved_names = encode_blendshapes(
        result.face_blendshapes[0] if result.face_blendshapes else [],
        blendshape_names,
    )
    return dense.reshape(-1).astype(np.float32), geometry, blendshape_vector, resolved_names


def extract_dataset_features(
    dataset_root: Path,
    *,
    landmarker_model: Path = DEFAULT_LANDMARKER_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force: bool = False,
) -> dict[str, object]:
    """Extract and cache all features derived from the local dataset."""
    if cache_path.exists() and not force:
        return joblib.load(cache_path)

    dataset_root = dataset_root.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[FeatureRecord] = []
    blendshape_names: list[str] | None = None

    with create_face_landmarker(landmarker_model, output_face_blendshapes=True) as detector:
        for split, label, path in _image_files(dataset_root):
            extracted = _extract_features_for_image(path, detector, blendshape_names)
            if extracted is None:
                records.append(
                    FeatureRecord(
                        split=split,
                        label=label,
                        image_path=str(path),
                        detected=False,
                        dense_features=None,
                        geometric_features=None,
                        blendshape_features=None,
                    )
                )
                continue

            dense, geometry, blendshape_vector, blendshape_names = extracted
            records.append(
                FeatureRecord(
                    split=split,
                    label=label,
                    image_path=str(path),
                    detected=True,
                    dense_features=dense,
                    geometric_features=geometry,
                    blendshape_features=blendshape_vector,
                )
            )

    if blendshape_names is None:
        raise RuntimeError("No valid faces were detected in the dataset.")

    payload = {
        "dataset_root": str(dataset_root),
        "created_at": datetime.now(UTC).isoformat(),
        "blendshape_names": blendshape_names,
        "records": records,
        "geometric_feature_names": GEOMETRIC_FEATURE_NAMES,
    }
    joblib.dump(payload, cache_path)
    return payload


def _coverage_from_records(records: list[FeatureRecord]) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for split in ("train", "val", "test"):
        split_summary: dict[str, dict[str, int]] = {}
        for label in CLASS_NAMES:
            label_records = [record for record in records if record.split == split and record.label == label]
            detected = sum(1 for record in label_records if record.detected)
            split_summary[label] = {
                "total": len(label_records),
                "detected": detected,
                "discarded": len(label_records) - detected,
            }
        summary[split] = split_summary
    return summary


def _dataset_arrays(
    records: list[FeatureRecord],
    feature_set: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    label_map = labels_to_index()
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "val", "test"):
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        for record in records:
            if record.split != split or not record.detected:
                continue
            assert record.dense_features is not None
            assert record.geometric_features is not None
            assert record.blendshape_features is not None
            dense = record.dense_features.reshape(478, 3)
            vector = build_feature_vector(dense, record.geometric_features, record.blendshape_features, feature_set)
            vectors.append(vector)
            labels.append(label_map[record.label])
        arrays[split] = (np.stack(vectors).astype(np.float32), np.array(labels, dtype=np.int32))
    return arrays


def _candidate_specs(random_state: int = 42) -> list[CandidateSpec]:
    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.98, svd_solver="full")),
            ("classifier", LogisticRegression(max_iter=2500, random_state=random_state)),
        ]
    )
    return [
        CandidateSpec(
            name="logistic_regression_pca",
            feature_set="dense_geometry",
            estimator_factory=lambda: clone(logistic_pipeline),
        ),
        CandidateSpec(
            name="random_forest_blendshape",
            feature_set="geometry_blendshape",
            estimator_factory=lambda: RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        CandidateSpec(
            name="hist_gradient_boosting_blendshape",
            feature_set="geometry_blendshape",
            estimator_factory=lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=8,
                max_iter=300,
                random_state=random_state,
            ),
        ),
    ]


def _latency_ms(estimator: object, features: np.ndarray, iterations: int = 5) -> float:
    sample = features[: min(len(features), 64)]
    start = time.perf_counter()
    for _ in range(iterations):
        estimator.predict_proba(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / (len(sample) * iterations)) * 1000.0


def _fit_and_rank_candidates(records: list[FeatureRecord]) -> list[dict[str, object]]:
    ranking: list[dict[str, object]] = []
    for candidate in _candidate_specs():
        arrays = _dataset_arrays(records, candidate.feature_set)
        x_train, y_train = arrays["train"]
        x_val, y_val = arrays["val"]
        estimator = candidate.estimator_factory()
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_val)
        probabilities = estimator.predict_proba(x_val)
        ranking.append(
            {
                "name": candidate.name,
                "feature_set": candidate.feature_set,
                "val_macro_f1": float(f1_score(y_val, y_pred, average="macro")),
                "val_accuracy": float(accuracy_score(y_val, y_pred)),
                "val_latency_ms_per_sample": float(_latency_ms(estimator, x_val)),
                "estimator": estimator,
                "val_confidence_mean": float(np.max(probabilities, axis=1).mean()),
                "spec": candidate,
            }
        )
    ranking.sort(
        key=lambda item: (
            item["val_macro_f1"],
            item["val_accuracy"],
            -item["val_latency_ms_per_sample"],
        ),
        reverse=True,
    )
    return ranking


def _plot_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_title("Matriz de confusao - conjunto de teste")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_confidence_histogram(confidences: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(confidences, bins=np.linspace(0.0, 1.0, 11), color="#1f77b4", edgecolor="black")
    axis.set_title("Distribuicao de confianca - conjunto de teste")
    axis.set_xlabel("Confianca")
    axis.set_ylabel("Quantidade de frames")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _report_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Relatorio de validacao",
        "",
        f"- Modelo vencedor: `{summary['selected_model']}`",
        f"- Feature set: `{summary['feature_set']}`",
        f"- Accuracy (teste): `{summary['test_accuracy']:.4f}`",
        f"- Macro-F1 (teste): `{summary['test_macro_f1']:.4f}`",
        f"- Threshold de confianca: `{summary['confidence_threshold']:.2f}`",
        "",
        "## Cobertura de deteccao",
    ]
    coverage = summary["coverage"]
    for split, labels in coverage.items():
        lines.append(f"### {split}")
        for label, counts in labels.items():
            lines.append(
                f"- {label}: {counts['detected']}/{counts['total']} detectadas, {counts['discarded']} descartadas"
            )
        lines.append("")
    lines.append("## Selecionador de modelos")
    for candidate in summary["model_selection"]:
        lines.append(
            f"- {candidate['name']}: macro-F1={candidate['val_macro_f1']:.4f}, "
            f"accuracy={candidate['val_accuracy']:.4f}, "
            f"latencia={candidate['val_latency_ms_per_sample']:.4f} ms/amostra"
        )
    lines.append("")
    lines.append("## Classification report")
    lines.append("```json")
    lines.append(json.dumps(summary["classification_report"], indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


def train_and_validate(
    dataset_root: Path,
    *,
    landmarker_model: Path = DEFAULT_LANDMARKER_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    output_model_path: Path,
    report_dir: Path = DEFAULT_REPORT_DIR,
    force_feature_extraction: bool = False,
) -> dict[str, object]:
    """Train the candidate models, evaluate them and persist the winner."""
    feature_payload = extract_dataset_features(
        dataset_root,
        landmarker_model=landmarker_model,
        cache_path=cache_path,
        force=force_feature_extraction,
    )
    records = feature_payload["records"]
    coverage = _coverage_from_records(records)
    ranking = _fit_and_rank_candidates(records)
    winner = ranking[0]

    arrays = _dataset_arrays(records, winner["feature_set"])
    x_train = np.concatenate([arrays["train"][0], arrays["val"][0]], axis=0)
    y_train = np.concatenate([arrays["train"][1], arrays["val"][1]], axis=0)
    x_test, y_test = arrays["test"]

    calibrated = CalibratedClassifierCV(
        estimator=winner["spec"].estimator_factory(),
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(x_train, y_train)
    test_probabilities = calibrated.predict_proba(x_test)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    test_macro_f1 = float(f1_score(y_test, test_predictions, average="macro"))
    matrix = confusion_matrix(y_test, test_predictions, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(y_test, test_predictions, target_names=CLASS_NAMES, output_dict=True)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    try:
        portable_landmarker_model = str(landmarker_model.resolve().relative_to(Path.cwd()))
    except ValueError:
        portable_landmarker_model = str(landmarker_model)

    artifact = {
        "model": calibrated,
        "labels": CLASS_NAMES,
        "feature_set": winner["feature_set"],
        "blendshape_names": feature_payload["blendshape_names"],
        "geometric_feature_names": GEOMETRIC_FEATURE_NAMES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "smoothing_window": SMOOTHING_WINDOW,
        "selected_model": winner["name"],
        "landmarker_model": portable_landmarker_model,
        "created_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(artifact, output_model_path, compress=3)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_root": str(dataset_root.resolve()),
        "feature_cache": str(cache_path.resolve()),
        "landmarker_model": str(landmarker_model.resolve()),
        "selected_model": winner["name"],
        "feature_set": winner["feature_set"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "coverage": coverage,
        "model_selection": [
            {
                key: value
                for key, value in candidate.items()
                if key not in {"estimator", "spec"}
            }
            for candidate in ranking
        ],
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confidence_summary": {
            "mean": float(np.mean(np.max(test_probabilities, axis=1))),
            "median": float(np.median(np.max(test_probabilities, axis=1))),
            "min": float(np.min(np.max(test_probabilities, axis=1))),
            "max": float(np.max(np.max(test_probabilities, axis=1))),
        },
    }

    metrics_path = report_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "validation_report.md").write_text(_report_markdown(summary), encoding="utf-8")
    _plot_confusion_matrix(matrix, CLASS_NAMES, report_dir / "confusion_matrix.png")
    _plot_confidence_histogram(np.max(test_probabilities, axis=1), report_dir / "confidence_histogram.png")
    return summary
