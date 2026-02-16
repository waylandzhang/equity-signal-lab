import pytest
import pandas as pd
from src.data import load_data, calculate_returns, check_stationarity, train_test_split_by_date


def test_load_data_returns_dataframe(data_path):
    df = load_data(data_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST']
    assert len(df) > 0
    assert df['Ticker'].nunique() == 5


def test_calculate_returns_adds_columns(raw_df):
    result = calculate_returns(raw_df)

    assert 'overnight_return' in result.columns
    assert 'intraday_return' in result.columns
    # Row count preserved (corporate actions are NaN'd, not dropped)
    assert len(result) == len(raw_df)

    # First row per ticker should have NaN overnight return
    for ticker in result['Ticker'].unique():
        ticker_df = result[result['Ticker'] == ticker]
        assert pd.isna(ticker_df.iloc[0]['overnight_return'])
        assert not pd.isna(ticker_df.iloc[0]['intraday_return'])


def test_check_stationarity_on_returns(df_with_returns, first_ticker):
    ticker_data = df_with_returns[df_with_returns['Ticker'] == first_ticker]['overnight_return'].dropna()
    result = check_stationarity(ticker_data, f'{first_ticker} overnight')

    assert 'adf_stat' in result
    assert 'p_value' in result
    assert 'is_stationary' in result
    assert result['is_stationary'] in (True, False)


def test_train_test_split_by_date(df_with_returns):
    # Split at the midpoint of the dataset
    mid_date = df_with_returns['DATE'].quantile(0.5)
    mid_str = mid_date.strftime('%Y-%m-%d')

    train, test = train_test_split_by_date(df_with_returns, mid_str)

    assert len(train) > 0
    assert len(test) > 0
    assert train['DATE'].max() < pd.to_datetime(mid_str)
    assert test['DATE'].min() >= pd.to_datetime(mid_str)
