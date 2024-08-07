import numpy as np
import pytest
from facsimile.model import FACSIMILE
from facsimile.eval import (
    calculate_score,
    evaluate_facsimile,
    FACSIMILEOptimiser,
)
from sklearn.datasets import make_regression
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


def test_evaluate_facsimile_with_valid_data():
    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    alphas = (0.01, 0.01)

    metrics, n_items = evaluate_facsimile(X_train, y_train, X_val, y_val, alphas)

    assert isinstance(metrics, dict)
    assert isinstance(metrics["score"], float)
    assert isinstance(metrics["r2"], np.ndarray)
    assert isinstance(n_items, np.int64)
    assert metrics["r2"].shape == (2,)


def test_evaluate_facsimile_with_pandas_data():
    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    alphas = (0.01, 0.01)

    metrics, n_items = evaluate_facsimile(X_train, y_train, X_val, y_val, alphas)

    assert isinstance(metrics, dict)
    assert isinstance(metrics["score"], float)
    assert isinstance(metrics["r2"], np.ndarray)
    assert isinstance(n_items, np.int64)
    assert metrics["r2"].shape == (2,)


def test_evaluate_facsimile_with_fitting_failure():
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 3)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20, 3)
    alphas = (1000, 1000, 1000)

    metrics, n_items = evaluate_facsimile(X_train, y_train, X_val, y_val, alphas)

    print(metrics)

    assert isinstance(metrics, dict)
    assert np.isnan(metrics["score"])
    assert np.all(np.isnan(metrics["r2"]))
    assert np.isnan(n_items)


def test_evaluate_facsimile_with_custom_metric():
    x, y = make_regression(n_samples=400, n_features=10, n_targets=2, random_state=42)

    # Scale the data
    x = scale(x)
    y = scale(y)

    x = pd.DataFrame(x)
    y = pd.DataFrame(y)

    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    alphas = (0.01, 0.01)

    metrics, n_items = evaluate_facsimile(
        X_train,
        y_train,
        X_val,
        y_val,
        alphas,
        additional_metrics={"mse": mean_squared_error},
    )

    assert isinstance(metrics, dict)
    assert "mse" in metrics
    assert isinstance(metrics["mse"], float)
    assert not np.isnan(metrics["mse"])


def test_calculate_score_with_numpy_array():
    r2 = np.array([0.8, 0.85, 0.9])
    n_included_items = 5
    n_features = 10
    expected_score = np.min(r2) * (1 - n_included_items / n_features)
    assert calculate_score(r2, n_included_items, n_features) == expected_score


def test_calculate_score_with_list():
    r2 = [0.8, 0.85, 0.9]
    n_included_items = 5
    n_features = 10
    expected_score = np.min(np.array(r2)) * (1 - n_included_items / n_features)
    assert calculate_score(r2, n_included_items, n_features) == expected_score


def test_calculate_score_with_single_value():
    r2 = [0.9]
    n_included_items = 1
    n_features = 10
    expected_score = np.min(np.array(r2)) * (1 - n_included_items / n_features)
    assert calculate_score(r2, n_included_items, n_features) == expected_score


def test_calculate_score_with_zero_included_items():
    r2 = [0.8, 0.85, 0.9]
    n_included_items = 0
    n_features = 10
    expected_score = np.min(np.array(r2))
    assert calculate_score(r2, n_included_items, n_features) == expected_score


def test_calculate_score_with_all_included_items():
    r2 = [0.8, 0.85, 0.9]
    n_included_items = 10
    n_features = 10
    expected_score = np.min(np.array(r2)) * 0
    assert calculate_score(r2, n_included_items, n_features) == expected_score


def test_get_best_classifier():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method
    best_classifier = optimiser.get_best_classifier()

    # Assertions
    assert isinstance(best_classifier, FACSIMILE)
    assert all(best_classifier.alphas == [0.03, 0.04])


def test_get_best_classifier_with_custom_metric():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser(additional_metrics={"mse": mean_squared_error})

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "mse": [0.75, 0.82, 0.88],
            "min_mse": [0.75, 0.82, 0.88],
            "max_mse": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method with a custom metric
    best_classifier = optimiser.get_best_classifier(
        metric="max_mse", highest_best=False
    )

    # Assertions
    assert isinstance(best_classifier, FACSIMILE)
    assert all(best_classifier.alphas == [0.01, 0.02])


def test_get_best_classifier_max_items_with_valid_results():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2, 3],
            "r2_Factor1": [0.8, 0.85, 0.9, 0.95],
            "r2_Factor2": [0.75, 0.82, 0.88, 0.9],
            "min_r2": [0.75, 0.82, 0.88, 0.9],
            "scores": [0.75, 0.82, 0.88, 0.9],
            "n_items": [5, 6, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03, 0.05],
            "alpha_Factor2": [0.02, 0.03, 0.04, 0.06],
        }
    )

    # Call the method with max_items = 6
    best_classifier = optimiser.get_best_classifier_max_items(max_items=6)

    # Assertions
    assert isinstance(best_classifier, FACSIMILE)
    assert all(best_classifier.alphas == [0.03, 0.04])


def test_get_best_classifier_max_items_with_no_results():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Call the method without setting up any results
    with pytest.raises(ValueError):
        optimiser.get_best_classifier_max_items(max_items=6)


def test_get_best_classifier_max_items_with_invalid_metric():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method with an invalid metric
    with pytest.raises(ValueError):
        optimiser.get_best_classifier_max_items(max_items=6, metric="invalid_metric")


def test_get_best_classifier_max_items_with_custom_metric():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method with max_items = 6 and metric = "min_r2"
    best_classifier = optimiser.get_best_classifier_max_items(
        max_items=6, metric="scores"
    )

    # Assertions
    assert isinstance(best_classifier, FACSIMILE)
    assert all(best_classifier.alphas == [0.02, 0.03])


def test_get_best_classifier_n_items_with_valid_results():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method with n_items = 6
    best_classifier = optimiser.get_best_classifier_n_items(n_items=6)

    # Assertions
    assert isinstance(best_classifier, FACSIMILE)
    assert all(best_classifier.alphas == [0.02, 0.03])


def test_get_best_classifier_n_items_does_not_exist():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call with a number of items that does not exist
    with pytest.raises(ValueError):
        _ = optimiser.get_best_classifier_n_items(n_items=8)


def test_get_best_classifier_n_items_with_no_results():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Call the method without setting up any results
    with pytest.raises(ValueError):
        optimiser.get_best_classifier_n_items(n_items=6)


def test_get_best_classifier_n_items_with_invalid_metric():
    # Create a mock instance of FACSIMILEOptimiser
    optimiser = FACSIMILEOptimiser()

    # Set up some mock results
    optimiser.results_ = pd.DataFrame(
        {
            "run": [0, 1, 2],
            "r2_Factor1": [0.8, 0.85, 0.9],
            "r2_Factor2": [0.75, 0.82, 0.88],
            "min_r2": [0.75, 0.82, 0.88],
            "scores": [0.75, 0.82, 0.88],
            "n_items": [5, 6, 7],
            "alpha_Factor1": [0.01, 0.02, 0.03],
            "alpha_Factor2": [0.02, 0.03, 0.04],
        }
    )

    # Call the method with an invalid metric
    with pytest.raises(ValueError):
        optimiser.get_best_classifier_n_items(n_items=6, metric="invalid_metric")
