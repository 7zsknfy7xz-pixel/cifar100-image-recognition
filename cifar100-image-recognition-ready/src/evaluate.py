from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_accuracy = {}
    for idx, name in enumerate(class_names):
        total = cm[idx].sum()
        per_class_accuracy[f"acc_{name}"] = float(cm[idx, idx] / total) if total else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy,
    }


def top_confusions(cm: np.ndarray, class_names: list[str], top_n: int = 50) -> list[dict]:
    rows = []
    for true_idx, true_name in enumerate(class_names):
        true_total = int(cm[true_idx].sum())
        if true_total == 0:
            continue
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            rows.append(
                {
                    "true_label": true_name,
                    "predicted_label": pred_name,
                    "count": count,
                    "true_total": true_total,
                    "rate_in_true_class": count / true_total,
                }
            )
    rows.sort(key=lambda row: (row["count"], row["rate_in_true_class"]), reverse=True)
    return rows[:top_n]
