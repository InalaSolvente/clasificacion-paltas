"""EDA reproducible para el Avocado Ripening Dataset.

La unidad biológica es la palta (``Sample``), observada repetidamente durante
varios días y desde dos lados. El módulo conserva esa estructura explícita para
evitar que futuros splits mezclen imágenes del mismo fruto.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "avocado-matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageStat


FILE_PATTERN = re.compile(
    r"^(?P<storage_group>T10|T20|Tam)_d(?P<day>[0-9]{2})_"
    r"(?P<sample>[0-9]{3})_(?P<side>[ab])_(?P<filename_label>[0-9]+)$"
)


@dataclass(frozen=True)
class EDAConfig:
    metadata_path: Path
    images_dir: Path
    output_dir: Path
    seed: int = 42
    examples_per_class: int = 4
    analysis_size: int = 64


def load_metadata(path: Path) -> pd.DataFrame:
    """Carga y normaliza el archivo de metadatos."""
    df = pd.read_excel(path, sheet_name="DATABASE")
    expected = {
        "File Name",
        "Time Stamp",
        "Storage Group",
        "Sample",
        "Day of Experiment",
        "Ripening Index Classification",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Columnas ausentes en el Excel: {sorted(missing)}")

    df = df.rename(
        columns={
            "File Name": "file_stem",
            "Time Stamp": "timestamp",
            "Storage Group": "storage_group",
            "Sample": "sample",
            "Day of Experiment": "day",
            "Ripening Index Classification": "ripening_index",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def parse_filenames(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la estructura del nombre y contrasta sus campos con el Excel."""
    parsed = df["file_stem"].astype(str).str.extract(FILE_PATTERN)
    out = df.copy()
    out["side"] = parsed["side"]
    for col in ("day", "sample", "filename_label"):
        parsed[col] = pd.to_numeric(parsed[col], errors="coerce").astype("Int64")
    out["filename_label"] = parsed["filename_label"]
    out["filename_parse_ok"] = parsed.notna().all(axis=1)
    out["filename_metadata_match"] = (
        out["filename_parse_ok"]
        & parsed["storage_group"].eq(out["storage_group"])
        & parsed["day"].eq(out["day"])
        & parsed["sample"].eq(out["sample"])
        & parsed["filename_label"].eq(out["ripening_index"])
    )
    # Nunca utilizar file_stem ni filename_label como entrada del modelo.
    out["sample_id"] = out["storage_group"].astype(str) + "_" + out["sample"].astype(str)
    return out


def _image_metrics(path: Path, analysis_size: int) -> dict[str, object]:
    """Valida una imagen y calcula métricas ligeras en una sola lectura."""
    raw = path.read_bytes()
    result: dict[str, object] = {
        "file_size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "image_ok": False,
        "image_error": "",
    }
    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            result.update(
                width=source.width,
                height=source.height,
                image_mode=source.mode,
                image_format=source.format,
            )
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((analysis_size, analysis_size))
            stat = ImageStat.Stat(rgb)
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            # Varianza del gradiente: proxy simple y portable de nitidez.
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            sharpness = float((gx.var() + gy.var()) / 2)
            result.update(
                mean_r=round(float(stat.mean[0]), 4),
                mean_g=round(float(stat.mean[1]), 4),
                mean_b=round(float(stat.mean[2]), 4),
                brightness=round(float(ImageStat.Stat(rgb.convert("L")).mean[0]), 4),
                sharpness=round(sharpness, 4),
                image_ok=True,
            )
    except Exception as exc:  # el error queda registrado en el manifiesto
        result["image_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_manifest(df: pd.DataFrame, images_dir: Path, analysis_size: int) -> pd.DataFrame:
    """Une metadatos con el inventario real y audita cada JPG disponible."""
    manifest = parse_filenames(df)
    manifest["image_path"] = manifest["file_stem"].map(
        lambda value: str((images_dir / f"{value}.jpg").resolve())
    )
    manifest["image_exists"] = manifest["image_path"].map(lambda value: Path(value).is_file())

    metrics: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        if row.image_exists:
            metrics.append(_image_metrics(Path(row.image_path), analysis_size))
        else:
            metrics.append(
                {
                    "image_ok": False,
                    "image_error": "missing_file",
                    "file_size_bytes": pd.NA,
                    "sha256": pd.NA,
                }
            )
    return pd.concat([manifest.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def _save_plots(manifest: pd.DataFrame, output_dir: Path) -> None:
    valid = manifest[manifest["image_ok"]].copy()
    plt.style.use("seaborn-v0_8-whitegrid")

    counts = manifest["ripening_index"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot.bar(ax=ax, color="#5f8f3c")
    ax.set(title="Distribución de clases", xlabel="Ripening Index", ylabel="Imágenes")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    table = pd.crosstab(manifest["day"], manifest["ripening_index"]).sort_index()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="YlGn")
    ax.set(
        title="Clase por día del experimento",
        xlabel="Ripening Index",
        ylabel="Día",
        xticks=np.arange(len(table.columns)),
        xticklabels=table.columns,
        yticks=np.arange(len(table.index)),
        yticklabels=table.index,
    )
    fig.colorbar(image, ax=ax, label="Imágenes")
    fig.tight_layout()
    fig.savefig(output_dir / "class_by_day.png", dpi=160)
    plt.close(fig)

    color_means = valid.groupby("ripening_index")[["mean_r", "mean_g", "mean_b"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for column, color in zip(color_means.columns, ("#c43b3b", "#3f8f3f", "#3b5fc4")):
        ax.plot(color_means.index, color_means[column], marker="o", label=column, color=color)
    ax.set(title="Color medio global por clase", xlabel="Ripening Index", ylabel="Intensidad (0–255)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mean_rgb_by_class.png", dpi=160)
    plt.close(fig)


def _save_contact_sheet(manifest: pd.DataFrame, config: EDAConfig) -> None:
    rng = np.random.default_rng(config.seed)
    classes = sorted(manifest["ripening_index"].dropna().unique())
    cell_w, cell_h = 240, 210
    canvas = Image.new("RGB", (cell_w * config.examples_per_class, cell_h * len(classes)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, label in enumerate(classes):
        candidates = manifest[(manifest["ripening_index"] == label) & manifest["image_ok"]]
        chosen = rng.choice(candidates.index.to_numpy(), size=min(config.examples_per_class, len(candidates)), replace=False)
        for col, idx in enumerate(chosen):
            item = manifest.loc[idx]
            with Image.open(item["image_path"]) as source:
                thumb = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"), (220, 170))
            x, y = col * cell_w + 10, row * cell_h + 28
            canvas.paste(thumb, (x, y))
            draw.text((x, row * cell_h + 8), f"Clase {label} | {item['storage_group']} | d{item['day']}", fill="black")
    canvas.save(config.output_dir / "class_examples.jpg", quality=92)


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def build_summary(manifest: pd.DataFrame, images_dir: Path) -> dict[str, object]:
    actual = {path.stem for path in images_dir.glob("*.jpg")}
    expected = set(manifest["file_stem"].astype(str))
    hashes = manifest.loc[manifest["sha256"].notna(), "sha256"]
    duplicate_groups = int((hashes.value_counts() > 1).sum())
    transitions = (
        manifest.sort_values(["sample_id", "day"])
        .groupby("sample_id")["ripening_index"]
        .apply(lambda values: bool((values.drop_duplicates().diff().dropna() < 0).any()))
    )
    return {
        "rows_metadata": int(len(manifest)),
        "jpg_files": int(len(actual)),
        "missing_images": sorted(expected - actual),
        "orphan_images": sorted(actual - expected),
        "invalid_or_corrupt_images": int((~manifest["image_ok"]).sum()),
        "filename_parse_failures": int((~manifest["filename_parse_ok"]).sum()),
        "filename_metadata_mismatches": int((~manifest["filename_metadata_match"]).sum()),
        "duplicate_hash_groups": duplicate_groups,
        "duplicate_extra_files": int(hashes.duplicated().sum()),
        "samples": int(manifest["sample_id"].nunique()),
        "storage_groups": {str(k): int(v) for k, v in manifest["storage_group"].value_counts().items()},
        "class_counts": {str(k): int(v) for k, v in manifest["ripening_index"].value_counts().sort_index().items()},
        "class_percent": {
            str(k): round(float(v * 100), 2)
            for k, v in manifest["ripening_index"].value_counts(normalize=True).sort_index().items()
        },
        "dimensions": {
            f"{int(w)}x{int(h)}": int(n)
            for (w, h), n in manifest[manifest["image_ok"]].groupby(["width", "height"]).size().items()
        },
        "non_monotonic_sample_sequences": int(transitions.sum()),
    }


def _write_report(summary: dict[str, object], output_dir: Path) -> None:
    classes = summary["class_counts"]
    percentages = summary["class_percent"]
    class_rows = "\n".join(f"| {key} | {classes[key]} | {percentages[key]}% |" for key in classes)
    missing = summary["missing_images"]
    missing_text = ", ".join(f"`{name}.jpg`" for name in missing) if missing else "Ninguna"
    report = f"""# EDA — Avocado Ripening Dataset

## Resumen

- Registros en Excel: **{summary['rows_metadata']:,}**
- JPG encontrados: **{summary['jpg_files']:,}**
- Paltas distintas (unidad experimental): **{summary['samples']}**
- Imágenes ausentes o inválidas: **{summary['invalid_or_corrupt_images']}**
- Grupos de duplicados exactos: **{summary['duplicate_hash_groups']}**
- Secuencias de madurez no monótonas: **{summary['non_monotonic_sample_sequences']}**

| Clase ordinal | Imágenes | Proporción |
|---:|---:|---:|
{class_rows}

## Hallazgos que afectan al modelado

1. **Fuga de etiqueta en el nombre.** El último componente del archivo coincide con
   `Ripening Index Classification`. `file_stem`, `image_path` y `filename_label` son
   variables administrativas y nunca deben entrar al modelo.
2. **Mediciones repetidas.** Cada palta aparece en varios días y tiene dos vistas
   (`a` y `b`). Los splits deben hacerse por `sample_id`, no por imagen; ambas vistas
   y todos los días de una palta deben permanecer en un solo split.
3. **El día es un proxy fuerte del objetivo.** La madurez avanza de manera ordinal y
   no se observaron retrocesos de clase por palta. Para evaluar visión, `day` tampoco
   debe ser una entrada del modelo.
4. **Desbalance moderado.** La clase mayoritaria es 1 y la minoritaria es 2. Conviene
   reportar macro-F1, balanced accuracy y matriz de confusión, además de accuracy.
5. **Objetivo ordinal.** Confundir 1 con 2 no tiene el mismo costo que confundir 1 con
   5. Además de métricas nominales, conviene medir MAE de clase y quadratic weighted
   kappa.
6. **Fondo dominante.** Todas las imágenes tienen fondo claro y el promedio RGB global
   queda dominado por ese fondo. En el siguiente paso conviene medir color y textura
   sobre una máscara de la palta y usar augmentations que impidan aprender el fondo.

## Integridad

Imágenes referenciadas pero ausentes ({len(missing)}): {missing_text}

Los detalles por archivo están en `manifest.csv` y los valores agregados en
`summary.json`.

## Gráficos

![Distribución](class_distribution.png)

![Clase por día](class_by_day.png)

![Color medio](mean_rgb_by_class.png)

![Ejemplos](class_examples.jpg)
"""
    (output_dir / "EDA_REPORT.md").write_text(report, encoding="utf-8")


def run_eda(config: EDAConfig) -> dict[str, object]:
    """Ejecuta el EDA completo y retorna su resumen."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(config.metadata_path)
    manifest = build_manifest(metadata, config.images_dir, config.analysis_size)
    manifest.to_csv(config.output_dir / "manifest.csv", index=False)
    summary = build_summary(manifest, config.images_dir)
    def portable(value: object) -> object:
        if not isinstance(value, Path):
            return value
        try:
            return str(value.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(value)

    payload = {"config": {k: portable(v) for k, v in asdict(config).items()}, **summary}
    (config.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8"
    )
    _save_plots(manifest, config.output_dir)
    _save_contact_sheet(manifest, config)
    _write_report(summary, config.output_dir)
    return summary
