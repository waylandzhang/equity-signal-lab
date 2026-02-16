import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from typing import Dict, Any
import os

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Optional date array for per-date cross-sectional IC

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Direction accuracy: % of times sign is correct
    direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_true))

    # Baseline: always-long direction accuracy (fraction of positive returns)
    baseline_accuracy = np.mean(np.array(y_true) > 0)

    # Information Coefficient: per-date cross-sectional Spearman, then mean
    if dates is not None:
        date_ics = []
        for d in np.unique(dates):
            mask = dates == d
            if mask.sum() >= 3:  # need ≥3 obs for meaningful rank correlation
                ic_d, _ = spearmanr(y_true[mask], y_pred[mask])
                if not np.isnan(ic_d):
                    date_ics.append(ic_d)
        ic = np.mean(date_ics) if date_ics else float('nan')
    else:
        ic, _ = spearmanr(y_true, y_pred)

    # Error statistics
    errors = y_pred - y_true

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'ic': ic,
        'mean_error': np.mean(errors),
        'std_error': np.std(errors, ddof=1)
    }


def calculate_sharpe_ratio(
    returns: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Simplified strategy: long if predicted return > 0, short if < 0.

    Args:
        returns: Strategy returns
        annualization_factor: 252 for daily data

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns, ddof=1) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        results: {'model_name': metrics_dict, ...}

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    df = pd.DataFrame(results).T
    return df.round(4)


def plot_predictions_vs_actuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actuals",
    save_path: str = None
):
    """Scatter plot of predicted vs actual values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    # 45-degree reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')

    # R²
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residuals_over_time(
    dates: pd.Series,
    residuals: np.ndarray,
    title: str = "Residuals Over Time",
    save_path: str = None
):
    """Time series plot of residuals."""
    plt.figure(figsize=(12, 5))
    plt.plot(dates, residuals, alpha=0.6, linewidth=0.8)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero error')

    # Rolling mean and std
    residuals_series = pd.Series(residuals, index=dates)
    rolling_mean = residuals_series.rolling(window=20, min_periods=1).mean()
    rolling_std = residuals_series.rolling(window=20, min_periods=1).std()

    plt.plot(dates, rolling_mean, 'g-', label='20-day rolling mean', linewidth=2)
    plt.fill_between(dates, rolling_mean - rolling_std, rolling_mean + rolling_std,
                     alpha=0.2, color='green', label='±1 std')

    plt.xlabel('Date')
    plt.ylabel('Prediction Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: str = None
):
    """Bar chart of feature importance."""
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_performance(
    fold_scores: list,
    title: str = "CV Performance",
    save_path: str = None
):
    """Plot CV fold scores over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(fold_scores, marker='o')
    plt.axhline(y=np.mean(fold_scores), color='r', linestyle='--', label=f'Mean: {np.mean(fold_scores):.4f}')
    plt.xlabel('CV Fold')
    plt.ylabel('Score (R²)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
