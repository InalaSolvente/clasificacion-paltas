"""Baseline clásico con descriptores visuales interpretables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .training import classification_metrics


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Separa aproximadamente la palta del fondo blanco del estudio."""
    distance_from_white = np.linalg.norm(255.0 - rgb.astype(np.float32), axis=2)
    mask = distance_from_white > 45
    if mask.mean() < 0.03:
        mask = rgb.mean(axis=2) < 245
    return mask


def extract_handcrafted_features(image_path: Path | str, size: int = 128) -> dict[str, float]:
    """Extrae color, forma y textura sin usar metadatos ni nombre de archivo."""
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((size, size))
    rgb = np.asarray(image, dtype=np.uint8)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    mask = _foreground_mask(rgb)
    if not mask.any():
        raise ValueError(f"No se pudo segmentar la palta: {image_path}")

    features: dict[str, float] = {"foreground_fraction": float(mask.mean())}
    for space_name, values in (("rgb", rgb), ("hsv", hsv)):
        for channel in range(3):
            pixels = values[:, :, channel][mask].astype(np.float32)
            for statistic, value in (
                ("mean", pixels.mean()),
                ("std", pixels.std()),
                ("q25", np.quantile(pixels, 0.25)),
                ("median", np.quantile(pixels, 0.50)),
                ("q75", np.quantile(pixels, 0.75)),
            ):
                features[f"{space_name}_{channel}_{statistic}"] = float(value)

    rows, columns = np.where(mask)
    height = max(int(rows.max() - rows.min() + 1), 1)
    width = max(int(columns.max() - columns.min() + 1), 1)
    features["bbox_aspect_ratio"] = width / height
    features["bbox_fill"] = float(mask.sum() / (height * width))

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    inner_x = mask[:, 1:] & mask[:, :-1]
    inner_y = mask[1:, :] & mask[:-1, :]
    features["texture_gradient_mean"] = float(
        np.mean(np.concatenate([gx[inner_x], gy[inner_y]]))
    )
    features["texture_gradient_std"] = float(
        np.std(np.concatenate([gx[inner_x], gy[inner_y]]))
    )
    return features


def build_handcrafted_feature_table(
    frame: pd.DataFrame,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Calcula una vez los descriptores y permite reutilizarlos entre folds."""
    identity = frame[["file_stem", "image_path", "ripening_index", "cv_fold"]].copy()
    if cache_path is not None and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(identity) and set(cached["file_stem"]) == set(identity["file_stem"]):
            # cv_fold puede cambiar con la semilla; se toma siempre del frame actual.
            feature_columns = [column for column in cached if column.startswith("feature_")]
            return identity.merge(cached[["file_stem", *feature_columns]], on="file_stem")

    rows = []
    for item in identity.itertuples(index=False):
        features = extract_handcrafted_features(item.image_path)
        rows.append({"file_stem": item.file_stem, **{f"feature_{k}": v for k, v in features.items()}})
    result = identity.merge(pd.DataFrame(rows), on="file_stem")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
    return result


def cross_validate_handcrafted_baseline(
    folded_frame: pd.DataFrame,
    cache_path: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evalúa regresión logística con los mismos folds agrupados."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Instala scikit-learn desde requirements-notebook.txt") from exc

    table = build_handcrafted_feature_table(folded_frame, cache_path)
    feature_columns = [column for column in table if column.startswith("feature_")]
    development = table[table["cv_fold"] >= 0]
    labels = sorted(development["ripening_index"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    rows = []
    for fold in sorted(development["cv_fold"].unique()):
        train = development[development["cv_fold"] != fold]
        validation = development[development["cv_fold"] == fold]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=-1
            ),
        )
        model.fit(train[feature_columns], train["ripening_index"])
        predictions = model.predict(validation[feature_columns])
        targets_index = validation["ripening_index"].map(label_to_index).to_numpy()
        predictions_index = pd.Series(predictions).map(label_to_index).to_numpy()
        metrics = classification_metrics(targets_index, predictions_index, len(labels))
        rows.append({"model": "handcrafted_logistic", "fold": int(fold), **metrics})
    return pd.DataFrame(rows)
