import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
import lightgbm as lgb
from typing import Dict, Any

def train_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Train Ridge regression model.

    Args:
        X: Feature matrix
        y: Target variable
        alpha: Regularization strength

    Returns:
        Dictionary with model and metadata
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    return {
        'model': model,
        'alpha': alpha,
        'intercept': model.intercept_,
        'coef': model.coef_,
        'n_features': X.shape[1]
    }


def train_elastic_net(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.0001,
    l1_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Train Elastic Net model.

    Args:
        X: Feature matrix
        y: Target variable
        alpha: Regularization strength
        l1_ratio: 0=Ridge, 1=Lasso, 0.5=balanced

    Returns:
        Dictionary with model and metadata
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    model.fit(X, y)

    # Count non-zero coefficients (selected features)
    n_selected = np.sum(model.coef_ != 0)

    return {
        'model': model,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'intercept': model.intercept_,
        'coef': model.coef_,
        'n_features': X.shape[1],
        'n_features_selected': n_selected
    }


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Train LightGBM model with strong regularization.

    Args:
        X: Feature matrix
        y: Target variable
        params: Optional model parameters

    Returns:
        Dictionary with model and metadata
    """
    default_params = {
        'max_depth': 3,
        'num_leaves': 7,
        'min_child_samples': 20,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'verbosity': -1,
        'random_state': 42
    }

    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X, y)

    return {
        'model': model,
        'params': default_params,
        'feature_importance': model.feature_importances_,
        'n_features': X.shape[1]
    }
