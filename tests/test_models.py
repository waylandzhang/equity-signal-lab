import pytest
import pandas as pd
import numpy as np
from src.features import engineer_features_v1
from src.models import train_ridge, train_elastic_net, train_lightgbm


@pytest.fixture
def sample_data(df_with_returns):
    """Prepare sample X, y for model testing."""
    df = engineer_features_v1(df_with_returns)
    feature_cols = [col for col in df.columns if col not in
                    ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST', 'overnight_return', 'intraday_return']]
    df_clean = df[feature_cols + ['overnight_return']].dropna()
    return df_clean[feature_cols], df_clean['overnight_return']


def test_train_ridge_returns_model(sample_data):
    X, y = sample_data

    result = train_ridge(X, y, alpha=1.0)

    assert 'model' in result
    assert 'alpha' in result
    assert 'intercept' in result
    assert 'coef' in result
    assert result['alpha'] == 1.0

    # Model should be able to predict
    predictions = result['model'].predict(X)
    assert len(predictions) == len(X)


def test_train_elastic_net_returns_model(sample_data):
    X, y = sample_data

    result = train_elastic_net(X, y, alpha=1.0, l1_ratio=0.5)

    assert 'model' in result
    assert 'l1_ratio' in result
    assert 'n_features_selected' in result
    assert result['l1_ratio'] == 0.5


def test_train_lightgbm_returns_model(sample_data):
    X, y = sample_data

    result = train_lightgbm(X, y)

    assert 'model' in result
    assert 'feature_importance' in result
    assert len(result['feature_importance']) == X.shape[1]
