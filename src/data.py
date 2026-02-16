import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate stock price data from Excel."""
    df = pd.read_excel(filepath)

    # Validate columns
    required_cols = ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST']
    if list(df.columns) != required_cols:
        raise ValueError(f"Expected columns {required_cols}, got {list(df.columns)}")

    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'DATE']).reset_index(drop=True)

    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Missing values found in data")

    # Check for duplicates
    if df.duplicated(subset=['Ticker', 'DATE']).any():
        raise ValueError("Duplicate ticker-date pairs found")

    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overnight and intraday returns for each ticker."""
    df = df.copy()

    # Calculate returns within each ticker group
    # Overnight: Open(t) / Close(t-1) - 1
    df['overnight_return'] = df.groupby('Ticker')['PX_LAST'].shift(1)
    df['overnight_return'] = (df['PX_OPEN'] / df['overnight_return']) - 1

    # Intraday: Close(t) / Open(t) - 1
    df['intraday_return'] = (df['PX_LAST'] / df['PX_OPEN']) - 1

    # NaN corporate-action outlier returns (stock splits, etc.)
    # Only NaN the affected return column — intraday may still be valid on split days
    for col in ['overnight_return', 'intraday_return']:
        extreme = df[col].abs() > 0.5
        if extreme.any():
            for idx in df[extreme].index:
                row = df.loc[idx]
                print(f"NaN corporate action: {row['Ticker']} {row['DATE'].date()} "
                      f"{col}={row[col]:.4f}")
            df.loc[extreme, col] = np.nan

    return df


def check_stationarity(series: pd.Series, name: str = 'series') -> Dict[str, any]:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())

    output = {
        'adf_stat': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05  # 5% significance
    }

    print(f"\n{name} Stationarity Test:")
    print(f"  ADF Statistic: {output['adf_stat']:.4f}")
    print(f"  p-value: {output['p_value']:.4f}")
    print(f"  Stationary: {output['is_stationary']}")

    return output


def train_test_split_by_date(df: pd.DataFrame, test_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data at specific date (not random) - all tickers split at same date."""
    test_date_pd = pd.to_datetime(test_date)
    train_df = df[df['DATE'] < test_date_pd]
    test_df = df[df['DATE'] >= test_date_pd]
    return train_df, test_df
