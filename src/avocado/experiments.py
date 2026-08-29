"""Registro liviano y auditable de resultados experimentales."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import ExperimentConfig


def save_experiment_config(config: ExperimentConfig, run_dir: Path) -> Path:
    """Guarda la configuración exacta de una corrida."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.json"
    path.write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def log_result(
    config: ExperimentConfig,
    metrics: Mapping[str, float],
    registry_path: Path,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    """Agrega una corrida como una línea JSON sin sobrescribir historial."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.to_dict(),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts or {}),
    }
    with registry_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
