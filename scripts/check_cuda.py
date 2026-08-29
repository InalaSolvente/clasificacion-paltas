"""Verifica CUDA usando exactamente el Python que ejecuta este comando."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avocado.training import cuda_diagnostics, resolve_device  # noqa: E402


if __name__ == "__main__":
    details = cuda_diagnostics()
    print(json.dumps(details, indent=2))
    device = resolve_device("cuda")
    print(f"CUDA lista para entrenar en {device}")
