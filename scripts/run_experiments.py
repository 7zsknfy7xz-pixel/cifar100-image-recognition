from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical import train_hog_svm, train_pca_svm
from src.config import DATA_DIR, DEFAULT_SEED, OUTPUT_DIR
from src.data import (
    dataset_to_numpy,
    get_image_datasets,
    get_tensor_datasets,
    make_loader,
    set_seed,
)
from src.evaluate import classification_metrics, top_confusions
from src.models import build_model
from src.plots import plot_confusion_matrix, plot_prediction_grid, plot_training_curve
from src.train import get_device, predict, train_resnet

CLASSICAL_METHODS = {"pca_svm", "hog_svm"}
DEEP_METHODS = {"resnet18", "convnext_tiny"}
SUPPORTED_METHODS = CLASSICAL_METHODS | DEEP_METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIFAR image classification experiments.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--methods", nargs="+", default=["pca_svm", "hog_svm", "resnet18"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means full training set.")
    parser.add_argument("--test-limit", type=int, default=0, help="0 means full test set.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, help="Input image size for deep models. ConvNeXt defaults to 224.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", nargs="+", type=int, help="Run repeated experiments and save mean/std metrics.")
    parser.add_argument("--tune-classical", action="store_true", help="Run small grid searches for PCA/SVM and HOG/SVM.")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights.")
    return parser.parse_args()


def save_method_outputs(
    method: str,
    test_images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
    seed: int,
    artifact_suffix: str = "",
) -> dict:
    metrics = classification_metrics(y_true, y_pred, class_names)
    artifact_name = f"{method}{artifact_suffix}"
    dense_confusion_values = len(class_names) <= 30
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        f"{method} Confusion Matrix",
        output_dir / f"confusion_matrix_{artifact_name}.png",
        include_values=dense_confusion_values,
    )
    if not dense_confusion_values:
        pd.DataFrame(top_confusions(metrics["confusion_matrix"], class_names)).to_csv(
            output_dir / f"top_confusions_{artifact_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    plot_prediction_grid(
        test_images,
        y_true,
        y_pred,
        class_names,
        output_dir / f"sample_predictions_{artifact_name}.png",
        f"{method} Sample Predictions",
        only_errors=False,
    )
    plot_prediction_grid(
        test_images,
        y_true,
        y_pred,
        class_names,
        output_dir / f"misclassified_{artifact_name}.png",
        f"{method} Misclassified Examples",
        only_errors=True,
    )
    row = {
        "seed": seed,
        "method": method,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
    }
    row.update(metrics["per_class_accuracy"])
    return row


def save_training_history(history: dict[str, list[float]], output_path: Path, seed: int, method: str) -> None:
    rows = []
    epochs = len(history["train_loss"])
    for idx in range(epochs):
        rows.append(
            {
                "seed": seed,
                "method": method,
                "epoch": idx + 1,
                "train_loss": history["train_loss"][idx],
                "train_acc": history["train_acc"][idx],
                "val_loss": history["val_loss"][idx],
                "val_acc": history["val_acc"][idx],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def run_one_seed(args: argparse.Namespace, seed: int, multi_seed: bool) -> list[dict]:
    set_seed(seed)
    methods = set(args.methods)
    unknown_methods = methods - SUPPORTED_METHODS
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {sorted(unknown_methods)}")

    rows = []
    artifact_suffix = f"_seed{seed}" if multi_seed else ""

    if methods & CLASSICAL_METHODS:
        train_set, test_set, class_names = get_image_datasets(
            args.dataset,
            args.data_dir,
            args.train_limit,
            args.test_limit,
            seed,
        )
        train_images, train_labels = dataset_to_numpy(train_set)
        test_images, test_labels = dataset_to_numpy(test_set)

        if "pca_svm" in methods:
            print(f"Running pca_svm on {args.dataset} with seed={seed}...")
            pred = train_pca_svm(train_images, train_labels, test_images, tune=args.tune_classical)
            rows.append(
                save_method_outputs(
                    "pca_svm",
                    test_images,
                    test_labels,
                    pred,
                    class_names,
                    args.output_dir,
                    seed,
                    artifact_suffix,
                )
            )

        if "hog_svm" in methods:
            print(f"Running hog_svm on {args.dataset} with seed={seed}...")
            pred = train_hog_svm(train_images, train_labels, test_images, tune=args.tune_classical)
            rows.append(
                save_method_outputs(
                    "hog_svm",
                    test_images,
                    test_labels,
                    pred,
                    class_names,
                    args.output_dir,
                    seed,
                    artifact_suffix,
                )
            )

    for method in sorted(methods & DEEP_METHODS):
        print(f"Running {method} on {args.dataset} with seed={seed} on {get_device()}...")
        train_set, test_set, class_names = get_tensor_datasets(
            args.dataset,
            args.data_dir,
            args.train_limit,
            args.test_limit,
            seed,
            method,
            args.image_size,
        )
        train_loader = make_loader(train_set, args.batch_size, shuffle=True)
        test_loader = make_loader(test_set, args.batch_size, shuffle=False)
        model = build_model(method, num_classes=len(class_names), pretrained=args.pretrained)
        checkpoint_name = f"best_{method}{artifact_suffix}.pt"
        model, history = train_resnet(
            model,
            train_loader,
            test_loader,
            args.epochs,
            args.lr,
            args.output_dir / "checkpoints" / checkpoint_name,
            model_name=method,
        )
        plot_training_curve(history, args.output_dir / f"training_curve_{method}{artifact_suffix}.png", method)
        save_training_history(history, args.output_dir / f"training_history_{method}{artifact_suffix}.csv", seed, method)
        y_true, y_pred = predict(model, test_loader)

        _, image_test_set, _ = get_image_datasets(args.dataset, args.data_dir, 0, args.test_limit, seed)
        test_images, _ = dataset_to_numpy(image_test_set)
        rows.append(
            save_method_outputs(
                method,
                test_images,
                np.asarray(y_true),
                np.asarray(y_pred),
                class_names,
                args.output_dir,
                seed,
                artifact_suffix,
            )
        )

    return rows


def save_metric_tables(rows: list[dict], output_dir: Path) -> None:
    metrics_df = pd.DataFrame(rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics to {metrics_path}")

    value_cols = [col for col in metrics_df.columns if col not in {"seed", "method"}]
    summary = metrics_df.groupby("method")[value_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.copy().reset_index()
    summary_path = output_dir / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved metric summary to {summary_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1
    rows = []
    for seed in seeds:
        rows.extend(run_one_seed(args, seed, multi_seed))

    if not rows:
        raise ValueError("No methods were selected. Use --methods pca_svm hog_svm resnet18.")

    save_metric_tables(rows, args.output_dir)


if __name__ == "__main__":
    main()
