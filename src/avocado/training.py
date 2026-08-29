"""Backend PyTorch común para comparar backbones preentrenados."""

from __future__ import annotations

import json
import random
import sys
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AugmentationConfig, ExperimentConfig
from .data import materialize_cross_validation_fold


MODEL_REGISTRY = {
    "resnet18": "resnet18.a1_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "dino_vits16": "vit_small_patch16_224.dino",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnetv2_s": "efficientnetv2_rw_s.ra2_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "dino_vitb16": "vit_base_patch16_224.dino",
}

# Perfil conservador para RTX 5060 Ti 16 GB con AMP. Todos alcanzan batch
# efectivo 64, aunque los modelos más grandes usan acumulación de gradientes.
MODEL_PROFILE_16GB: dict[str, dict[str, int]] = {
    "resnet50": {"image_size": 224, "batch_size": 64, "gradient_accumulation_steps": 1},
    "efficientnetv2_s": {"image_size": 288, "batch_size": 32, "gradient_accumulation_steps": 2},
    "convnext_small": {"image_size": 224, "batch_size": 32, "gradient_accumulation_steps": 2},
    "swin_tiny": {"image_size": 224, "batch_size": 32, "gradient_accumulation_steps": 2},
    "dino_vits16": {"image_size": 224, "batch_size": 64, "gradient_accumulation_steps": 1},
    "dino_vitb16": {"image_size": 224, "batch_size": 16, "gradient_accumulation_steps": 4},
}


def cuda_diagnostics() -> dict[str, object]:
    """Describe el intérprete y la compatibilidad CUDA del kernel actual."""
    torch, _, _, _ = _require_torch()
    available = bool(torch.cuda.is_available())
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def resolve_device(accelerator: str = "cuda") -> Any:
    """Resuelve el dispositivo y evita fallback silencioso a CPU."""
    torch, _, _, _ = _require_torch()
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA fue requerida pero este kernel no la detecta. "
                f"Diagnóstico: {cuda_diagnostics()}. Instala el wheel CUDA de "
                "PyTorch en ESTE kernel y reinícialo antes de entrenar."
            )
        return torch.device("cuda:0")
    if accelerator == "cpu":
        return torch.device("cpu")
    raise ValueError("accelerator debe ser 'cuda' o 'cpu'")


@dataclass
class TrainingResult:
    model_name: str
    best_epoch: int
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float] | None
    checkpoint_path: str
    history_path: str


def select_best_validation(
    results: list[TrainingResult], metric: str = "macro_f1"
) -> TrainingResult:
    """Selecciona una corrida usando exclusivamente una métrica de validación."""
    if not results:
        raise ValueError("No hay resultados para seleccionar")
    missing = [result.model_name for result in results if metric not in result.validation_metrics]
    if missing:
        raise ValueError(f"La métrica {metric} falta para: {missing}")
    return max(results, key=lambda result: result.validation_metrics[metric])


def cross_validate_experiment(
    config: ExperimentConfig,
    folded_frame: pd.DataFrame,
    run_dir: Path,
    *,
    num_workers: int = 4,
) -> tuple[list[TrainingResult], pd.DataFrame]:
    """Entrena todos los folds sin consultar test y retorna resultados agregados."""
    folds = sorted(value for value in folded_frame["cv_fold"].unique() if value >= 0)
    results: list[TrainingResult] = []
    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        fold_frame = materialize_cross_validation_fold(folded_frame, fold)
        fold_result = train_experiment(
            config,
            fold_frame,
            run_dir / f"fold_{fold}",
            evaluate_test=False,
            num_workers=num_workers,
        )
        results.append(fold_result)
        rows.append(
            {
                "model": config.model_name,
                "fold": fold,
                "best_epoch": fold_result.best_epoch,
                "history_path": fold_result.history_path,
                "checkpoint_path": fold_result.checkpoint_path,
                **fold_result.validation_metrics,
            }
        )
    table = pd.DataFrame(rows)
    return results, table


def load_loss_histories(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Combina los historiales JSON generados por todos los modelos y folds."""
    required = {"model", "fold", "history_path"}
    missing = required.difference(cv_results.columns)
    if missing:
        raise ValueError(f"Resultados CV incompletos; faltan {sorted(missing)}")
    histories: list[pd.DataFrame] = []
    for row in cv_results.itertuples(index=False):
        path = Path(row.history_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el historial: {path}")
        history = pd.read_json(path)
        history["model"] = row.model
        history["fold"] = int(row.fold)
        histories.append(history)
    return pd.concat(histories, ignore_index=True)


def plot_loss_evolution(
    cv_results: pd.DataFrame,
    output_path: Path | None = None,
) -> tuple[Any, Any]:
    """Grafica loss medio por época y su dispersión entre folds."""
    import matplotlib.pyplot as plt

    history = load_loss_histories(cv_results)
    models = list(dict.fromkeys(history["model"]))
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)
    for axis, model_name in zip(axes.ravel(), models):
        subset = history[history["model"] == model_name]
        for column, label, color in (
            ("train_loss", "Training loss", "#2c7fb8"),
            ("validation_loss", "Validation loss", "#d95f0e"),
        ):
            stats = subset.groupby("epoch")[column].agg(["mean", "std"])
            epochs = stats.index.to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0).to_numpy(dtype=float)
            axis.plot(epochs, mean, marker="o", linewidth=2, label=label, color=color)
            axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.16)
        axis.set(
            title=model_name,
            xlabel="Época",
            ylabel="Cross-entropy loss",
        )
        axis.legend()
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(models) :]:
        axis.axis("off")
    fig.suptitle("Evolución de loss: media ± desviación entre folds", fontsize=14)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=170, bbox_inches="tight")
    return fig, axes


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "Faltan dependencias de entrenamiento. Ejecuta: "
            "python -m pip install -r requirements-notebook.txt"
        ) from exc
    return torch, timm, DataLoader, (Dataset, transforms)


def seed_everything(seed: int) -> None:
    torch, _, _, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(
    image_size: int, augmentation: AugmentationConfig | None = None
) -> tuple[Any, Any]:
    """Augmentation moderado para train y transformación fija para evaluación."""
    _, _, _, (_, transforms) = _require_torch()
    augmentation = augmentation or AugmentationConfig()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if augmentation.enabled:
        train_steps = [
            transforms.RandomResizedCrop(
                image_size, scale=(augmentation.crop_scale_min, 1.0)
            ),
            transforms.RandomHorizontalFlip(augmentation.horizontal_flip_probability),
            transforms.RandomRotation(augmentation.rotation_degrees),
            transforms.ColorJitter(
                brightness=augmentation.brightness,
                contrast=augmentation.contrast,
                saturation=augmentation.saturation,
            ),
        ]
    else:
        train_steps = [transforms.Resize(int(image_size / 0.875)), transforms.CenterCrop(image_size)]
    train_transform = transforms.Compose(
        [*train_steps, transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, evaluation_transform


def plot_augmentation_preview(
    image_path: Path | str,
    image_size: int,
    augmentation: AugmentationConfig,
    examples: int = 6,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Muestra la imagen original y varias transformaciones de entrenamiento."""
    import matplotlib.pyplot as plt
    from PIL import Image

    torch, _, _, _ = _require_torch()
    torch.manual_seed(seed)
    transform, _ = build_transforms(image_size, augmentation)
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    fig, axes = plt.subplots(1, examples + 1, figsize=(3 * (examples + 1), 3))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    for index in range(examples):
        tensor = (transform(original) * std + mean).clamp(0, 1)
        axes[index + 1].imshow(tensor.permute(1, 2, 0).numpy())
        axes[index + 1].set_title(f"Aug. {index + 1}")
        axes[index + 1].axis("off")
    fig.tight_layout()
    return fig, axes


class AvocadoImageDataset:
    """Dataset serializable para workers de DataLoader en Windows."""

    def __init__(self, frame: pd.DataFrame, transform: Any, classes: tuple[int, ...]):
        self.paths = frame["image_path"].astype(str).tolist()
        self.labels = frame["ripening_index"].astype(int).tolist()
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        from PIL import Image

        torch, _, _, _ = _require_torch()
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        label = self.label_to_index[self.labels[index]]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> Any:
    """Construye cualquiera de los tres backbones con una cabeza equivalente."""
    _, timm, _, _ = _require_torch()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido: {model_name}. Opciones: {sorted(MODEL_REGISTRY)}")
    return timm.create_model(
        MODEL_REGISTRY[model_name], pretrained=pretrained, num_classes=num_classes
    )


def hardware_preflight(
    model_names: tuple[str, ...],
    profile: dict[str, dict[str, int]],
    num_classes: int = 5,
) -> pd.DataFrame:
    """Prueba forward/backward y mide VRAM antes de una corrida extensa."""
    torch, _, _, _ = _require_torch()
    device = resolve_device("cuda")
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        spec = profile[model_name]
        model = images = labels = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model = build_model(model_name, num_classes, pretrained=False).to(device)
            images = torch.randn(
                spec["batch_size"], 3, spec["image_size"], spec["image_size"], device=device
            )
            labels = torch.randint(0, num_classes, (spec["batch_size"],), device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            status, error = "ok", ""
        except RuntimeError as exc:
            status, error = "error", str(exc).splitlines()[0]
        rows.append(
            {
                "model": model_name,
                **spec,
                "effective_batch": spec["batch_size"] * spec["gradient_accumulation_steps"],
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
                "status": status,
                "error": error,
            }
        )
        del model, images, labels, loss
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> dict[str, float]:
    """Métricas nominales y ordinales, sin depender de sklearn."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    total = confusion.sum()
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    recalls = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(num_classes, dtype=float), where=confusion.sum(axis=1) != 0,
    )
    precisions = np.divide(
        np.diag(confusion), confusion.sum(axis=0),
        out=np.zeros(num_classes, dtype=float), where=confusion.sum(axis=0) != 0,
    )
    f1 = np.divide(
        2 * precisions * recalls, precisions + recalls,
        out=np.zeros(num_classes, dtype=float), where=(precisions + recalls) != 0,
    )
    mae = float(np.abs(targets - predictions).mean())

    weights = np.square(np.subtract.outer(np.arange(num_classes), np.arange(num_classes)))
    weights = weights / float((num_classes - 1) ** 2)
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / max(total, 1)
    denominator = float((weights * expected).sum())
    qwk = 1.0 - float((weights * confusion).sum()) / denominator if denominator else 0.0
    return {
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(float(recalls.mean()), 6),
        "macro_f1": round(float(f1.mean()), 6),
        "mae": round(mae, 6),
        "qwk": round(qwk, 6),
    }


def _evaluate(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, dict[str, float]]:
    torch, _, _, _ = _require_torch()
    model.eval()
    losses: list[float] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            targets.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    y_true, y_pred = np.concatenate(targets), np.concatenate(predictions)
    metrics = classification_metrics(y_true, y_pred, model.num_classes)
    return float(np.mean(losses)), metrics


def evaluate_checkpoint(
    config: ExperimentConfig,
    split_frame: pd.DataFrame,
    checkpoint_path: Path | str,
    *,
    split: str = "test",
    num_workers: int = 4,
) -> dict[str, float]:
    """Evalúa un checkpoint elegido por validación sin reentrenarlo."""
    torch, _, DataLoader, _ = _require_torch()
    if split not in {"validation", "test"}:
        raise ValueError("Solo se permite evaluar validation o test")
    device = resolve_device(config.accelerator)
    _, transform = build_transforms(config.image_size, config.augmentation)
    subset = split_frame[split_frame["split"] == split]
    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        AvocadoImageDataset(subset, transform, config.data.classes),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        **loader_options,
    )
    model = build_model(config.model_name, len(config.data.classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    criterion = torch.nn.CrossEntropyLoss()
    _, metrics = _evaluate(model, loader, criterion, device)
    return metrics


def train_experiment(
    config: ExperimentConfig,
    split_frame: pd.DataFrame,
    run_dir: Path,
    *,
    evaluate_test: bool = False,
    num_workers: int = 4,
) -> TrainingResult:
    """Entrena un modelo y selecciona por macro-F1 de validación.

    ``evaluate_test`` permanece desactivado durante la comparación de modelos.
    Debe habilitarse una sola vez para el modelo final elegido.
    """
    torch, _, DataLoader, _ = _require_torch()
    from tqdm.auto import tqdm

    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(f"model_name debe ser uno de {sorted(MODEL_REGISTRY)}")
    seed_everything(config.split.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.accelerator)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(
        f"Dispositivo: {device} | "
        f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} | "
        f"AMP: {config.mixed_precision and device.type == 'cuda'}"
    )
    train_transform, evaluation_transform = build_transforms(
        config.image_size, config.augmentation
    )
    loaders = {}
    for split in ("train", "validation", "test"):
        subset = split_frame[split_frame["split"] == split]
        transform = train_transform if split == "train" else evaluation_transform
        loader_options: dict[str, Any] = {}
        if num_workers > 0:
            loader_options.update(persistent_workers=True, prefetch_factor=2)
        loaders[split] = DataLoader(
            AvocadoImageDataset(subset, transform, config.data.classes),
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            **loader_options,
        )

    model = build_model(config.model_name, len(config.data.classes)).to(device)
    counts = split_frame.loc[split_frame["split"] == "train", "ripening_index"].value_counts()
    class_weights = torch.tensor(
        [1.0 / counts.get(label, 1) for label in config.data.classes],
        dtype=torch.float32, device=device,
    )
    class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float | int]] = []
    best_score, best_epoch, stale_epochs = -np.inf, 0, 0
    checkpoint_path = run_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        train_loader = loaders["train"]
        for batch_index, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"{config.model_name} {epoch}/{config.epochs}"), start=1
        ):
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                raw_loss = criterion(model(images), labels)
                loss = raw_loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(raw_loss.item()))

        validation_loss, validation_metrics = _evaluate(
            model, loaders["validation"], criterion, device
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
        }
        history.append(row)
        score = validation_metrics["macro_f1"]
        if score > best_score:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, validation_metrics = _evaluate(model, loaders["validation"], criterion, device)
    test_metrics = None
    if evaluate_test:
        _, test_metrics = _evaluate(model, loaders["test"], criterion, device)
    return TrainingResult(
        model_name=config.model_name,
        best_epoch=best_epoch,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
    )
