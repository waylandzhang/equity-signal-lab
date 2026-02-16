import pytest
import pandas as pd
from src.data import load_data, calculate_returns
from src.cv import time_series_cv_split


def test_time_series_cv_respects_constraints(df_with_returns):
    min_train = 21
    max_train = 100  # Use smaller max for testing

    splits = list(time_series_cv_split(df_with_returns, min_train_days=min_train, max_train_days=max_train))

    # Should have splits
    assert len(splits) > 0

    # Check first split has min_train days
    first_train_idx, first_test_idx = splits[0]
    train_dates = df_with_returns.loc[first_train_idx, 'DATE'].nunique()
    assert train_dates >= min_train

    # Test should have data
    assert len(first_test_idx) > 0


def test_cv_no_lookahead(df_with_returns):
    """Ensure test data comes after train data."""
    splits = list(time_series_cv_split(df_with_returns, min_train_days=21, max_train_days=100))

    for train_idx, test_idx in splits[:5]:  # Check first 5
        train_max_date = df_with_returns.loc[train_idx, 'DATE'].max()
        test_min_date = df_with_returns.loc[test_idx, 'DATE'].min()
        assert train_max_date < test_min_date
