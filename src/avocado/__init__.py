"""Herramientas para el framework de clasificación de madurez de paltas."""

from .eda import EDAConfig, run_eda
from .config import (
    AugmentationConfig,
    CrossValidationConfig,
    DataConfig,
    ExperimentConfig,
    SplitConfig,
)
from .data import (
    assert_no_group_leakage,
    assert_view_pairs,
    assign_cross_validation_folds,
    cross_validation_summary,
    grouped_split,
    load_experiment_data,
    materialize_cross_validation_fold,
    split_summary,
)
from .experiments import log_result, save_experiment_config
from .baselines import cross_validate_handcrafted_baseline, extract_handcrafted_features
from .training import (
    MODEL_REGISTRY,
    MODEL_PROFILE_16GB,
    TrainingResult,
    classification_metrics,
    cross_validate_experiment,
    cuda_diagnostics,
    evaluate_checkpoint,
    hardware_preflight,
    load_loss_histories,
    plot_loss_evolution,
    plot_augmentation_preview,
    resolve_device,
    select_best_validation,
    train_experiment,
)

__all__ = [
    "DataConfig",
    "AugmentationConfig",
    "CrossValidationConfig",
    "EDAConfig",
    "ExperimentConfig",
    "SplitConfig",
    "assert_no_group_leakage",
    "assert_view_pairs",
    "assign_cross_validation_folds",
    "cross_validation_summary",
    "cross_validate_handcrafted_baseline",
    "extract_handcrafted_features",
    "grouped_split",
    "load_experiment_data",
    "materialize_cross_validation_fold",
    "log_result",
    "MODEL_REGISTRY",
    "MODEL_PROFILE_16GB",
    "run_eda",
    "save_experiment_config",
    "split_summary",
    "TrainingResult",
    "classification_metrics",
    "cross_validate_experiment",
    "cuda_diagnostics",
    "evaluate_checkpoint",
    "hardware_preflight",
    "load_loss_histories",
    "plot_loss_evolution",
    "plot_augmentation_preview",
    "resolve_device",
    "select_best_validation",
    "train_experiment",
]
