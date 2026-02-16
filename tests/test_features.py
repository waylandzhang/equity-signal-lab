import pytest
import pandas as pd
from src.data import load_data, calculate_returns
from src.features import engineer_features_v1, engineer_features_v2, engineer_features_v3


def test_engineer_features_v1_no_lag_features(df_with_returns):
    result = engineer_features_v1(df_with_returns)

    # Should have calendar features
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'is_month_end' in result.columns
    assert 'is_quarter_end' in result.columns

    # Should have ticker dummies (N-1 = 4 for 5 tickers)
    ticker_dummies = [col for col in result.columns if col.startswith('ticker_')]
    assert len(ticker_dummies) == 4

    # Should NOT have any lag features
    assert not any('lag' in col for col in result.columns)
    assert not any('rolling' in col for col in result.columns)
    assert not any('volatility' in col for col in result.columns)


def test_engineer_features_v2_adds_minimal_temporal(df_with_returns):
    result = engineer_features_v2(df_with_returns)

    # Should have all v1 features
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns

    # Should have basket lag features
    assert 'basket_return_overnight_lag1' in result.columns
    assert 'basket_return_intraday_lag1' in result.columns

    # Should have rolling volatility
    assert 'volatility_5d_overnight' in result.columns
    assert 'volatility_5d_intraday' in result.columns

    # First row should have NaN for lag features
    assert pd.isna(result.iloc[0]['basket_return_overnight_lag1'])


def test_v2_no_lookahead_bias(df_with_returns, first_ticker):
    """Ensure lag features don't leak future information."""
    result = engineer_features_v2(df_with_returns)

    # For any row, basket_lag1 should use data from previous date
    ticker_df = result[result['Ticker'] == first_ticker].reset_index(drop=True)

    # Row 1 (index 1) basket_lag1 should NOT equal row 1's overnight_return
    # (it should be from row 0 or earlier)
    if len(ticker_df) > 1 and not pd.isna(ticker_df.loc[1, 'basket_return_overnight_lag1']):
        assert ticker_df.loc[1, 'basket_return_overnight_lag1'] != ticker_df.loc[1, 'overnight_return']


def test_engineer_features_v3_adds_full_temporal(df_with_returns):
    result = engineer_features_v3(df_with_returns)

    # Should have all v2 features
    assert 'basket_return_overnight_lag1' in result.columns

    # Should have rolling means
    assert 'rolling_mean_10d_overnight' in result.columns
    assert 'rolling_mean_20d_overnight' in result.columns

    # Should have rolling volatility
    assert 'volatility_10d_overnight' in result.columns
    assert 'volatility_20d_overnight' in result.columns

    # Should have momentum
    assert 'momentum_5d_overnight' in result.columns
    assert 'momentum_10d_overnight' in result.columns
