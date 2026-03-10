"""Train and validate the face emotion classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from face_emotion.constants import (
    DEFAULT_CACHE_PATH,
    DEFAULT_EMOTION_MODEL,
    DEFAULT_LANDMARKER_MODEL,
    DEFAULT_REPORT_DIR,
)
from face_emotion.training import train_and_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o classificador de emocoes faciais.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("eINTERFACE_2021_Image"),
        help="Diretorio local contendo os splits train/val/test.",
    )
    parser.add_argument(
        "--landmarker-model",
        type=Path,
        default=DEFAULT_LANDMARKER_MODEL,
        help="Asset face_landmarker.task utilizado na extracao.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Arquivo joblib para cache de features.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_EMOTION_MODEL,
        help="Arquivo joblib do classificador final.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Diretorio de saida para relatorios de validacao.",
    )
    parser.add_argument(
        "--force-feature-extraction",
        action="store_true",
        help="Forca a reextracao das features mesmo com cache existente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset nao encontrado em {args.dataset_root}.")
    summary = train_and_validate(
        dataset_root=args.dataset_root,
        landmarker_model=args.landmarker_model,
        cache_path=args.cache_path,
        output_model_path=args.output_model,
        report_dir=args.report_dir,
        force_feature_extraction=args.force_feature_extraction,
    )
    print(f"Modelo vencedor: {summary['selected_model']}")
    print(f"Accuracy (teste): {summary['test_accuracy']:.4f}")
    print(f"Macro-F1 (teste): {summary['test_macro_f1']:.4f}")
    print(f"Relatorio salvo em: {args.report_dir}")


if __name__ == "__main__":
    main()
