"""Tests for src/predict.py - predict_next_period interface."""

import pickle

import pytest
import pandas as pd
import numpy as np

from src.data import load_data, calculate_returns
from src.features import engineer_features_v1, engineer_features_v2
from src.models import train_ridge
from src.predict import predict_next_period


@pytest.fixture
def trained_model_path(tmp_path, data_path):
    """Train a ridge v1 model and save to a temp file."""
    df = load_data(data_path)
    df = calculate_returns(df)
    df = engineer_features_v1(df)

    exclude_cols = ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST',
                    'overnight_return', 'intraday_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_clean = df.dropna(subset=feature_cols + ['overnight_return'])

    result = train_ridge(df_clean[feature_cols], df_clean['overnight_return'])

    model_path = tmp_path / "test_ridge_v1.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)

    return str(model_path)


@pytest.fixture
def trained_model_v2_path(tmp_path, data_path):
    """Train a ridge v2 model and save to a temp file."""
    df = load_data(data_path)
    df = calculate_returns(df)
    df = engineer_features_v2(df)

    exclude_cols = ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST',
                    'overnight_return', 'intraday_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_clean = df.dropna(subset=feature_cols + ['overnight_return'])

    result = train_ridge(df_clean[feature_cols], df_clean['overnight_return'])

    model_path = tmp_path / "test_ridge_v2.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)

    return str(model_path)


def test_predict_returns_valid_result(trained_model_path, first_ticker, last_date, data_path):
    """predict_next_period returns expected dict structure for a valid ticker/date."""
    result = predict_next_period(
        ticker=first_ticker,
        date=last_date,
        model_path=trained_model_path,
        feature_version='v1',
        target='overnight_return',
        data_path=data_path
    )

    assert isinstance(result, dict)
    assert result['ticker'] == first_ticker
    assert result['date'] == last_date
    assert result['target'] == 'overnight_return'
    assert isinstance(result['predicted_return'], float)
    assert 'last_data_date' in result


def test_predict_v2_returns_valid_result(trained_model_v2_path, first_ticker, last_date, data_path):
    """predict_next_period works with v2 features."""
    result = predict_next_period(
        ticker=first_ticker,
        date=last_date,
        model_path=trained_model_v2_path,
        feature_version='v2',
        target='overnight_return',
        data_path=data_path
    )

    assert isinstance(result['predicted_return'], float)
    # Prediction should be a reasonable return (not absurdly large)
    assert abs(result['predicted_return']) < 1.0


def test_predict_all_tickers(trained_model_path, last_date, data_path):
    """predict_next_period works for every ticker in the dataset."""
    df = load_data(data_path)
    tickers = df['Ticker'].unique()

    for ticker in tickers:
        result = predict_next_period(
            ticker=ticker,
            date=last_date,
            model_path=trained_model_path,
            feature_version='v1',
            target='overnight_return',
            data_path=data_path
        )
        assert result['ticker'] == ticker
        assert isinstance(result['predicted_return'], float)


def test_predict_invalid_ticker_raises(trained_model_path, last_date, data_path):
    """predict_next_period raises ValueError for a non-existent ticker."""
    with pytest.raises(ValueError, match="No data found"):
        predict_next_period(
            ticker='FAKE_TICKER',
            date=last_date,
            model_path=trained_model_path,
            feature_version='v1',
            target='overnight_return',
            data_path=data_path
        )


def test_predict_date_before_data_raises(trained_model_path, first_ticker, data_path):
    """predict_next_period raises ValueError for a date before any data exists."""
    with pytest.raises(ValueError, match="No data found"):
        predict_next_period(
            ticker=first_ticker,
            date='2000-01-01',
            model_path=trained_model_path,
            feature_version='v1',
            target='overnight_return',
            data_path=data_path
        )


def test_predict_far_future_date_raises(trained_model_path, first_ticker, data_path):
    """predict_next_period rejects dates too far beyond available data."""
    with pytest.raises(ValueError, match="days beyond last available data"):
        predict_next_period(
            ticker=first_ticker,
            date='2099-12-31',
            model_path=trained_model_path,
            feature_version='v1',
            target='overnight_return',
            data_path=data_path
        )
