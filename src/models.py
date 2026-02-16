import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV
import lightgbm as lgb
from typing import Dict, Any

def train_ridge(X: pd.DataFrame, y: pd.Series,
                alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)) -> Dict[str, Any]:
    """
    Train Ridge regression with built-in alpha selection via leave-one-out CV.

    Args:
        X: Feature matrix
        y: Target variable
        alphas: Candidate regularization strengths

    Returns:
        Dictionary with model and metadata
    """
    model = RidgeCV(alphas=alphas)
    model.fit(X, y)

    return {
        'model': model,
        'alpha': float(model.alpha_),
        'intercept': model.intercept_,
        'coef': model.coef_,
        'n_features': X.shape[1]
    }


def train_elastic_net(
    X: pd.DataFrame,
    y: pd.Series,
    l1_ratio: tuple = (0.1, 0.5, 0.9),
    alphas: int = 20
) -> Dict[str, Any]:
    """
    Train Elastic Net with built-in alpha and l1_ratio selection via 3-fold CV.

    Args:
        X: Feature matrix
        y: Target variable
        l1_ratio: Candidate L1/L2 mixing ratios
        alphas: Number of alpha values to try per l1_ratio

    Returns:
        Dictionary with model and metadata
    """
    model = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
                         cv=3, max_iter=10000)
    model.fit(X, y)

    n_selected = np.sum(model.coef_ != 0)

    return {
        'model': model,
        'alpha': float(model.alpha_),
        'l1_ratio': float(model.l1_ratio_),
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
