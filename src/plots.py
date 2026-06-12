from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
    include_values: bool | None = None,
) -> None:
    configure_matplotlib()
    class_count = len(class_names)
    if include_values is None:
        include_values = class_count <= 30
    labels = class_names if class_count <= 30 else list(range(class_count))
    fig_size = (8, 7) if class_count <= 30 else (12, 10)
    fig, ax = plt.subplots(figsize=fig_size)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90 if class_count > 30 else 45,
        colorbar=class_count > 30,
        include_values=include_values,
    )
    ax.tick_params(axis="both", labelsize=5 if class_count > 30 else 9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_curve(history: dict[str, list[float]], output_path: Path, model_name: str = "ResNet18") -> None:
    configure_matplotlib()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = image.copy()
    if arr.max() <= 1.0:
        return np.clip(arr, 0, 1)
    return np.clip(arr / 255.0, 0, 1)


def plot_prediction_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    count: int = 16,
    only_errors: bool = False,
) -> None:
    configure_matplotlib()
    indices = np.arange(len(y_true))
    if only_errors:
        indices = indices[y_true != y_pred]
    indices = indices[:count]
    if len(indices) == 0:
        return

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.7 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, idx in zip(axes, indices):
        ax.imshow(_normalize_image(images[idx]))
        color = "green" if y_true[idx] == y_pred[idx] else "red"
        ax.set_title(
            f"T: {class_names[int(y_true[idx])]}\nP: {class_names[int(y_pred[idx])]}",
            color=color,
            fontsize=7 if len(class_names) > 30 else 9,
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
