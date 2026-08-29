"""Configuración centralizada de experimentos."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    """Rutas y reglas que definen el conjunto de datos usado."""

    manifest_path: Path
    valid_only: bool = True
    classes: tuple[int, ...] = (1, 2, 3, 4, 5)
    group_column: str = "sample_id"
    target_column: str = "ripening_index"
    stratify_column: str = "storage_group"


@dataclass(frozen=True)
class SplitConfig:
    """Partición reproducible a nivel de palta, nunca a nivel de imagen."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    seed: int = 42

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 for value in fractions):
            raise ValueError("Todas las fracciones deben ser positivas")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Las fracciones de split deben sumar 1")


@dataclass(frozen=True)
class CrossValidationConfig:
    """Cross-validation agrupada dentro del conjunto de desarrollo."""

    n_splits: int = 5
    seed: int = 42
    selection_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits debe ser al menos 2")


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentations aplicados exclusivamente durante entrenamiento."""

    enabled: bool = True
    crop_scale_min: float = 0.80
    horizontal_flip_probability: float = 0.50
    rotation_degrees: float = 10.0
    brightness: float = 0.12
    contrast: float = 0.12
    saturation: float = 0.08

    def __post_init__(self) -> None:
        if not 0 < self.crop_scale_min <= 1:
            raise ValueError("crop_scale_min debe estar en (0, 1]")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability debe estar en [0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración serializable de una corrida experimental."""

    name: str
    data: DataConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model_name: str = "pending"
    image_size: int = 224
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    accelerator: str = "cuda"
    mixed_precision: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.accelerator not in {"cuda", "cpu"}:
            raise ValueError("accelerator debe ser 'cuda' o 'cpu'")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps debe ser al menos 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"]["manifest_path"] = str(payload["data"]["manifest_path"])
        return payload
