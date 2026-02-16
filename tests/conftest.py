"""Shared test fixtures. Set EQUITY_DATA env var to test against a different dataset."""

import os
import pytest
import pandas as pd
from src.data import load_data, calculate_returns

DEFAULT_DATA_PATH = "data/raw/sample_tickers.xlsx"


@pytest.fixture
def data_path():
    """Data file path, overridable via EQUITY_DATA env var."""
    return os.environ.get("EQUITY_DATA", DEFAULT_DATA_PATH)


@pytest.fixture
def raw_df(data_path):
    """Loaded raw DataFrame."""
    return load_data(data_path)


@pytest.fixture
def df_with_returns(raw_df):
    """DataFrame with calculated returns."""
    return calculate_returns(raw_df)


@pytest.fixture
def first_ticker(raw_df):
    """First ticker in the dataset (for stationarity checks, etc.)."""
    return raw_df["Ticker"].iloc[0]


@pytest.fixture
def last_date(raw_df):
    """Last date in the dataset as string (for prediction tests)."""
    return raw_df["DATE"].max().strftime("%Y-%m-%d")
