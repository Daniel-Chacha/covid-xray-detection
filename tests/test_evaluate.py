import numpy as np
import pandas as pd
import pytest

from covid_xray.config import CLASS_NAMES
from covid_xray.evaluate import (
    bootstrap_ci,
    confusion,
    expected_calibration_error,
    macro_f1,
    per_class_metrics,
    restricted_pair_metrics,
    sensitivity_specificity,
    specificity_at_sensitivity,
    summarise_run,
)


@pytest.fixture
def perfect():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_prob = np.eye(4)[y_true] * 0.97 + 0.01
    return y_true, y_prob


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=400)
    logits = np.eye(4)[y_true] * 1.5 + rng.normal(size=(400, 4))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return y_true, exp / exp.sum(axis=1, keepdims=True)


def test_macro_f1_is_one_for_perfect_predictions(perfect):
    assert macro_f1(*perfect) == pytest.approx(1.0)


def test_per_class_metrics_has_a_row_per_class(noisy):
    table = per_class_metrics(*noisy, class_names=CLASS_NAMES)
    assert list(table.index) == list(CLASS_NAMES)
    assert {"precision", "recall", "f1", "support", "roc_auc", "pr_auc"} <= set(table.columns)


def test_sensitivity_and_specificity_are_perfect_on_perfect_input(perfect):
    sensitivity, specificity = sensitivity_specificity(*perfect, class_index=0)
    assert sensitivity == pytest.approx(1.0)
    assert specificity == pytest.approx(1.0)


def test_specificity_at_target_sensitivity_meets_the_target():
    rng = np.random.default_rng(1)
    y_true = np.concatenate([np.ones(100), np.zeros(300)])
    scores = np.concatenate([rng.normal(1.2, 1.0, 100), rng.normal(0.0, 1.0, 300)])

    specificity, threshold = specificity_at_sensitivity(y_true, scores, target=0.95)

    achieved = (scores[y_true == 1] >= threshold).mean()
    assert achieved >= 0.95 - 1e-9
    assert 0.0 <= specificity <= 1.0


def test_bootstrap_ci_brackets_the_point_estimate(noisy):
    y_true, y_prob = noisy
    point = macro_f1(y_true, y_prob)
    low, high = bootstrap_ci(y_true, y_prob, macro_f1, n_resamples=200, seed=42)
    assert low <= point <= high
    assert high - low > 0


def test_bootstrap_ci_is_reproducible(noisy):
    first = bootstrap_ci(*noisy, macro_f1, n_resamples=200, seed=42)
    second = bootstrap_ci(*noisy, macro_f1, n_resamples=200, seed=42)
    assert first == second


def test_confusion_matrix_is_square_and_totals_correctly(noisy):
    y_true, y_prob = noisy
    matrix = confusion(y_true, y_prob, CLASS_NAMES)
    assert matrix.shape == (4, 4)
    assert matrix.to_numpy().sum() == len(y_true)


def test_calibration_error_is_near_zero_for_confident_correct_predictions(perfect):
    assert expected_calibration_error(*perfect) < 0.10


def test_calibration_error_is_large_for_confidently_wrong_predictions():
    y_true = np.zeros(50, dtype=int)
    y_prob = np.tile([0.01, 0.97, 0.01, 0.01], (50, 1))
    assert expected_calibration_error(y_true, y_prob) > 0.5


def test_restricted_pair_uses_only_the_two_named_classes(noisy):
    y_true, y_prob = noisy
    result = restricted_pair_metrics(y_true, y_prob, pair=("COVID", "Lung_Opacity"))

    expected_n = int(((y_true == 0) | (y_true == 1)).sum())
    assert result["n"] == expected_n
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert result["pair"] == ("COVID", "Lung_Opacity")


def test_restricted_pair_ignores_the_other_classes_probability_mass():
    """Scores must be renormalised over the pair, not read off the 4-way softmax."""
    y_true = np.array([0, 1])
    # Both rows put most mass on Normal; within the pair, row 0 favours COVID.
    y_prob = np.array([[0.20, 0.05, 0.70, 0.05], [0.05, 0.20, 0.70, 0.05]])

    result = restricted_pair_metrics(y_true, y_prob, pair=("COVID", "Lung_Opacity"))

    assert result["accuracy"] == pytest.approx(1.0)


def test_summarise_run_returns_every_reported_field(noisy):
    summary = summarise_run("run1_raw", *noisy)
    assert summary["run"] == "run1_raw"
    for key in ("macro_f1", "macro_f1_ci", "covid_sensitivity", "covid_specificity",
                "covid_specificity_at_95_sensitivity", "ece", "per_class", "confusion",
                "restricted_pair"):
        assert key in summary
