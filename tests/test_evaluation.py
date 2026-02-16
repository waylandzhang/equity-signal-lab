import pytest
import numpy as np
from src.evaluation import calculate_metrics, compare_models, calculate_sharpe_ratio


def test_calculate_metrics_returns_dict():
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    y_pred = np.array([0.015, -0.015, 0.025, -0.005, 0.018])

    metrics = calculate_metrics(y_true, y_pred)

    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics
    assert 'direction_accuracy' in metrics
    assert 'ic' in metrics
    assert 'mean_error' in metrics
    assert 'std_error' in metrics

    # Direction accuracy should be between 0 and 1
    assert 0 <= metrics['direction_accuracy'] <= 1


def test_compare_models():
    results = {
        'Ridge': {'mse': 0.001, 'r2': 0.5, 'direction_accuracy': 0.55},
        'ElasticNet': {'mse': 0.0012, 'r2': 0.48, 'direction_accuracy': 0.53}
    }

    comparison = compare_models(results)

    assert len(comparison) == 2
    assert 'mse' in comparison.columns
    assert 'r2' in comparison.columns


def test_calculate_sharpe_ratio():
    returns = np.array([0.01, -0.005, 0.015, 0.008, -0.002])
    sharpe = calculate_sharpe_ratio(returns)

    assert isinstance(sharpe, float)
    # Sharpe ratio should be reasonable (wide range for test data)
    assert -50 < sharpe < 50
