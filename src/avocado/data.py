"""Carga, filtrado y splits sin fuga para experimentos de visión."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, SplitConfig


REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "image_ok",
    "sample_id",
    "storage_group",
    "ripening_index",
    "day",
    "side",
}


def load_experiment_data(config: DataConfig) -> pd.DataFrame:
    """Carga el manifiesto del EDA y aplica filtros declarativos."""
    manifest_path = Path(config.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Ejecuta primero: python scripts/run_eda.py"
        )
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"El manifiesto no contiene: {sorted(missing)}")

    if config.valid_only:
        frame = frame[frame["image_ok"].astype(bool)]
    frame = frame[frame[config.target_column].isin(config.classes)].copy()
    if frame.empty:
        raise ValueError("Los filtros dejaron el dataset vacío")
    frame[config.target_column] = frame[config.target_column].astype(int)
    frame["pair_id"] = (
        frame[config.group_column].astype(str)
        + "_d"
        + frame["day"].astype(int).astype(str).str.zfill(2)
    )
    return frame.reset_index(drop=True)


def _partition_counts(total: int, fractions: np.ndarray) -> np.ndarray:
    """Redondea fracciones conservando exactamente el total."""
    raw = fractions * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts))
    counts[order[:remainder]] += 1
    return counts


def grouped_split(
    frame: pd.DataFrame,
    data_config: DataConfig,
    split_config: SplitConfig,
) -> pd.DataFrame:
    """Asigna cada palta completa a un split.

    Se barajan muestras dentro de cada grupo de almacenamiento y se distribuyen
    según las fracciones solicitadas. Como todas las vistas y días comparten
    ``sample_id``, nunca cruzan entre train, validation y test.
    """
    group_col = data_config.group_column
    stratify_col = data_config.stratify_column
    for column in (group_col, stratify_col):
        if column not in frame:
            raise ValueError(f"Columna de split ausente: {column}")

    group_table = frame[[group_col, stratify_col]].drop_duplicates()
    if group_table[group_col].duplicated().any():
        raise ValueError(f"Un {group_col} pertenece a más de un {stratify_col}")

    names = np.array(["train", "validation", "test"])
    fractions = np.array(
        [split_config.train_fraction, split_config.validation_fraction, split_config.test_fraction]
    )
    rng = np.random.default_rng(split_config.seed)
    assignments: dict[object, str] = {}
    for _, subset in group_table.groupby(stratify_col, sort=True):
        groups = subset[group_col].to_numpy(copy=True)
        rng.shuffle(groups)
        counts = _partition_counts(len(groups), fractions)
        start = 0
        for name, count in zip(names, counts):
            for group in groups[start : start + count]:
                assignments[group] = str(name)
            start += int(count)

    result = frame.copy()
    result["split"] = result[group_col].map(assignments)
    if result["split"].isna().any():
        raise RuntimeError("Quedaron muestras sin asignar a un split")
    assert_no_group_leakage(result, group_col)
    return result


def assert_no_group_leakage(frame: pd.DataFrame, group_column: str = "sample_id") -> None:
    """Falla de inmediato si una palta aparece en más de un split."""
    if "split" not in frame:
        raise ValueError("El dataframe no contiene la columna split")
    leaked = frame.groupby(group_column)["split"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        raise AssertionError(f"Fuga detectada en {len(leaked)} grupos")


def assert_view_pairs(frame: pd.DataFrame) -> None:
    """Comprueba que los lados de una misma observación no crucen splits."""
    if "pair_id" not in frame:
        raise ValueError("El dataframe no contiene pair_id")
    crossed = frame.groupby("pair_id")["split"].nunique()
    if (crossed > 1).any():
        raise AssertionError(f"{int((crossed > 1).sum())} pares a/b cruzan splits")
    invalid_sides = frame.groupby("pair_id")["side"].agg(lambda x: set(x.dropna()))
    if not invalid_sides.map(lambda sides: sides == {"a", "b"}).all():
        raise AssertionError("Hay observaciones que no contienen exactamente los lados a y b")


def split_summary(
    frame: pd.DataFrame,
    target_column: str = "ripening_index",
    group_column: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna tamaños y distribución porcentual de clases por split."""
    sizes = frame.groupby("split").agg(
        images=(target_column, "size"),
        samples=(group_column, "nunique"),
    )
    sizes["image_percent"] = (sizes["images"] / len(frame) * 100).round(2)
    class_distribution = pd.crosstab(
        frame["split"], frame[target_column], normalize="index"
    ).mul(100).round(2)
    return sizes, class_distribution


def assign_cross_validation_folds(
    frame: pd.DataFrame,
    data_config: DataConfig,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Asigna folds al 80% de desarrollo y mantiene test completamente aislado.

    La asignación se hace por ``sample_id`` dentro de cada grupo de
    almacenamiento. Las filas de test reciben ``cv_fold = -1`` y nunca forman
    parte de entrenamiento ni validación cruzada.
    """
    if "split" not in frame:
        raise ValueError("Primero debes crear el split fijo train/validation/test")
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2")

    group_col = data_config.group_column
    stratify_col = data_config.stratify_column
    development = frame[frame["split"].isin(["train", "validation"])]
    group_table = development[[group_col, stratify_col]].drop_duplicates()
    if group_table[group_col].duplicated().any():
        raise ValueError(f"Un {group_col} pertenece a más de un {stratify_col}")

    rng = np.random.default_rng(seed)
    assignments: dict[object, int] = {}
    for _, subset in group_table.groupby(stratify_col, sort=True):
        groups = subset[group_col].to_numpy(copy=True)
        rng.shuffle(groups)
        # Round-robin mantiene el número de paltas por estrato casi idéntico.
        for position, group in enumerate(groups):
            assignments[group] = position % n_splits

    result = frame.copy()
    result["cv_fold"] = result[group_col].map(assignments).fillna(-1).astype(int)
    if not result.loc[result["split"] == "test", "cv_fold"].eq(-1).all():
        raise AssertionError("El conjunto test entró en cross-validation")
    if result.loc[result["split"] != "test", "cv_fold"].lt(0).any():
        raise AssertionError("Hay muestras de desarrollo sin fold")
    return result


def materialize_cross_validation_fold(frame: pd.DataFrame, fold: int) -> pd.DataFrame:
    """Convierte un fold asignado en columnas train/validation/test utilizables."""
    if "cv_fold" not in frame:
        raise ValueError("Primero ejecuta assign_cross_validation_folds")
    available = sorted(value for value in frame["cv_fold"].unique() if value >= 0)
    if fold not in available:
        raise ValueError(f"Fold {fold} no disponible; opciones: {available}")

    result = frame.copy()
    result["fixed_split"] = result["split"]
    development = result["cv_fold"] >= 0
    result.loc[development, "split"] = "train"
    result.loc[result["cv_fold"] == fold, "split"] = "validation"
    # Las filas cv_fold=-1 conservan el split test original.
    assert_no_group_leakage(result)
    assert_view_pairs(result)
    return result


def cross_validation_summary(
    frame: pd.DataFrame, group_column: str = "sample_id"
) -> pd.DataFrame:
    """Resume imágenes y paltas de validación en cada fold."""
    development = frame[frame["cv_fold"] >= 0]
    return development.groupby("cv_fold").agg(
        validation_images=("image_path", "size"),
        validation_samples=(group_column, "nunique"),
    )
