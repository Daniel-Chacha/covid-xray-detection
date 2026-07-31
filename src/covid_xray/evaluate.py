"""Metric suite.

Two decisions worth stating, both from spec section 6:

1. Every headline figure carries a bootstrap confidence interval. The Viral
   Pneumonia test slice is 200 images; a bare point estimate there implies a
   precision the data does not support.
2. Specificity is reported at a fixed 95% COVID sensitivity, not only at
   argmax. Argmax hides the trade-off that matters clinically — a false
   negative discharges an infectious patient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from covid_xray.config import CLASS_NAMES, CONTROL_PAIR


def predict_probabilities(model, dataset) -> np.ndarray:
    """Predicted class probabilities, in dataset order."""
    return np.asarray(model.predict(dataset, verbose=0), dtype=np.float64)


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(f1_score(y_true, y_prob.argmax(axis=1), average="macro", zero_division=0))


def per_class_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names=CLASS_NAMES
) -> pd.DataFrame:
    """Precision, recall, F1, support, one-vs-rest ROC-AUC and PR-AUC.

    PR-AUC sits alongside ROC-AUC because ROC-AUC is optimistic under the
    ~7:1 imbalance in this dataset.
    """
    y_pred = y_prob.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    roc_aucs, pr_aucs = [], []
    for index in range(len(class_names)):
        binary = (y_true == index).astype(int)
        if binary.min() == binary.max():  # class absent from this split
            roc_aucs.append(float("nan"))
            pr_aucs.append(float("nan"))
        else:
            roc_aucs.append(roc_auc_score(binary, y_prob[:, index]))
            pr_aucs.append(average_precision_score(binary, y_prob[:, index]))

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "roc_auc": roc_aucs,
            "pr_auc": pr_aucs,
        },
        index=list(class_names),
    )


def sensitivity_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, class_index: int
) -> tuple[float, float]:
    """One-vs-rest sensitivity and specificity at argmax."""
    y_pred = y_prob.argmax(axis=1)
    positives = y_true == class_index
    negatives = ~positives

    sensitivity = float((y_pred[positives] == class_index).mean()) if positives.any() else float("nan")
    specificity = float((y_pred[negatives] != class_index).mean()) if negatives.any() else float("nan")
    return sensitivity, specificity


def specificity_at_sensitivity(
    y_true_binary: np.ndarray, y_score: np.ndarray, target: float = 0.95
) -> tuple[float, float]:
    """Highest specificity achievable while holding sensitivity at or above `target`.

    Returns `(specificity, threshold)`. If the target is unreachable, returns
    the operating point with the highest sensitivity available.
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
    feasible = np.nonzero(tpr >= target)[0]
    index = feasible[0] if feasible.size else int(np.argmax(tpr))
    return float(1.0 - fpr[index]), float(thresholds[index])


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    statistic,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic of (y_true, y_prob)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = np.empty(n_resamples)

    for draw in range(n_resamples):
        positions = rng.integers(0, n, size=n)
        values[draw] = statistic(y_true[positions], y_prob[positions])

    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Standard binned ECE over top-1 confidence."""
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def confusion(y_true: np.ndarray, y_prob: np.ndarray, class_names=CLASS_NAMES) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_prob.argmax(axis=1), labels=range(len(class_names)))
    return pd.DataFrame(matrix, index=list(class_names), columns=list(class_names))


def restricted_pair_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, pair: tuple[str, str] = CONTROL_PAIR
) -> dict:
    """Metrics restricted to the two-class control pair.

    Both classes are adult, so the pediatric confound is absent. Scores are
    renormalised over the pair — reading them straight off the 4-way softmax
    would let a third class's probability mass decide the comparison.

    No retraining: this reuses the same trained model's predictions.
    """
    first, second = (CLASS_NAMES.index(name) for name in pair)
    selected = (y_true == first) | (y_true == second)

    if selected.sum() == 0:
        return {"pair": pair, "n": 0, "accuracy": float("nan"), "roc_auc": float("nan")}

    truth = (y_true[selected] == first).astype(int)
    pair_prob = y_prob[selected][:, [first, second]]
    score = pair_prob[:, 0] / np.clip(pair_prob.sum(axis=1), 1e-12, None)

    accuracy = float(((score >= 0.5).astype(int) == truth).mean())
    auc = float(roc_auc_score(truth, score)) if truth.min() != truth.max() else float("nan")

    return {"pair": pair, "n": int(selected.sum()), "accuracy": accuracy, "roc_auc": auc}


def summarise_run(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Everything the README reports for one run."""
    covid_index = CLASS_NAMES.index("COVID")
    sensitivity, specificity = sensitivity_specificity(y_true, y_prob, covid_index)
    spec_at_95, threshold = specificity_at_sensitivity(
        (y_true == covid_index).astype(int), y_prob[:, covid_index], target=0.95
    )

    return {
        "run": name,
        "macro_f1": macro_f1(y_true, y_prob),
        "macro_f1_ci": bootstrap_ci(y_true, y_prob, macro_f1),
        "covid_sensitivity": sensitivity,
        "covid_specificity": specificity,
        "covid_specificity_at_95_sensitivity": spec_at_95,
        "covid_threshold_at_95_sensitivity": threshold,
        "ece": expected_calibration_error(y_true, y_prob),
        "per_class": per_class_metrics(y_true, y_prob),
        "confusion": confusion(y_true, y_prob),
        "restricted_pair": restricted_pair_metrics(y_true, y_prob),
    }
