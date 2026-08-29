"""Punto de entrada del EDA del dataset de paltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avocado import EDAConfig, run_eda  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita el Avocado Ripening Dataset")
    parser.add_argument("--metadata", type=Path, default=ROOT / "Avocado Ripening Dataset.xlsx")
    parser.add_argument("--images", type=Path, default=ROOT / "Avocado Ripening Dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "eda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eda(
        EDAConfig(
            metadata_path=args.metadata,
            images_dir=args.images,
            output_dir=args.output,
            seed=args.seed,
        )
    )
    print(f"EDA listo: {summary['rows_metadata']} registros, {summary['jpg_files']} JPG")
    print(f"Informe: {(args.output / 'EDA_REPORT.md').resolve()}")


if __name__ == "__main__":
    main()
