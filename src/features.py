import pandas as pd
import numpy as np

def engineer_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering v1: Independence-first.
    Only ticker and calendar features, no temporal features.
    """
    df = df.copy()

    # Calendar features
    df['day_of_week'] = df['DATE'].dt.dayofweek  # 0=Monday, 4=Friday
    df['month'] = df['DATE'].dt.month

    # Month-end / quarter-end using calendar only (no lookahead)
    df['is_month_end'] = (df['DATE'] == df['DATE'] + pd.offsets.BMonthEnd(0)).astype(int)
    df['is_quarter_end'] = (df['DATE'] == df['DATE'] + pd.offsets.BQuarterEnd(0)).astype(int)

    # Ticker dummies (drop first to avoid multicollinearity)
    ticker_dummies = pd.get_dummies(df['Ticker'], prefix='ticker', drop_first=True)
    df = pd.concat([df, ticker_dummies], axis=1)

    return df


def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering v2: Minimal temporal features.
    Adds basket lag features and 5-day rolling volatility.
    """
    # Start with v1 features
    df = engineer_features_v1(df)

    # Sort by ticker and date for proper temporal order
    df = df.sort_values(['Ticker', 'DATE']).reset_index(drop=True)

    # Basket-wide lag features (cross-sectional)
    temp = df.copy()
    temp['basket_overnight'] = df.groupby('DATE')['overnight_return'].transform('mean')
    temp['basket_intraday'] = df.groupby('DATE')['intraday_return'].transform('mean')

    df['basket_return_overnight_lag1'] = temp.groupby('Ticker')['basket_overnight'].shift(1)
    df['basket_return_intraday_lag1'] = temp.groupby('Ticker')['basket_intraday'].shift(1)

    # Rolling volatility (per ticker, 5-day window)
    df['volatility_5d_overnight'] = df.groupby('Ticker')['overnight_return'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std().shift(1)
    )
    df['volatility_5d_intraday'] = df.groupby('Ticker')['intraday_return'].transform(
        lambda x: x.rolling(window=5, min_periods=2).std().shift(1)
    )

    return df


def engineer_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering v3: Full temporal features.
    Adds 10/20-day rolling stats and momentum.
    Warning: May overfit with only 519 days of data.
    """
    # Start with v2 features
    df = engineer_features_v2(df)

    # Rolling means and volatility per ticker
    for target, prefix in [('overnight_return', 'overnight'), ('intraday_return', 'intraday')]:
        # Rolling means (10-day, 20-day)
        df[f'rolling_mean_10d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: x.rolling(window=10, min_periods=5).mean().shift(1)
        )
        df[f'rolling_mean_20d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: x.rolling(window=20, min_periods=10).mean().shift(1)
        )

        # Rolling volatility (10-day, 20-day)
        df[f'volatility_10d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: x.rolling(window=10, min_periods=5).std().shift(1)
        )
        df[f'volatility_20d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: x.rolling(window=20, min_periods=10).std().shift(1)
        )

        # Momentum: cumulative returns
        df[f'momentum_5d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: ((1 + x).rolling(window=5, min_periods=3).apply(lambda y: y.prod() - 1, raw=True)).shift(1)
        )
        df[f'momentum_10d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: ((1 + x).rolling(window=10, min_periods=5).apply(lambda y: y.prod() - 1, raw=True)).shift(1)
        )
        df[f'momentum_20d_{prefix}'] = df.groupby('Ticker')[target].transform(
            lambda x: ((1 + x).rolling(window=20, min_periods=10).apply(lambda y: y.prod() - 1, raw=True)).shift(1)
        )

    return df
