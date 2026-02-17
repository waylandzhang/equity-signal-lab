import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

FEATURE_FUNCTIONS = None  # lazy-loaded to avoid circular imports


def _get_feature_functions():
    global FEATURE_FUNCTIONS
    if FEATURE_FUNCTIONS is None:
        from src.features import engineer_features_v1, engineer_features_v2, engineer_features_v3
        FEATURE_FUNCTIONS = {
            'v1': engineer_features_v1,
            'v2': engineer_features_v2,
            'v3': engineer_features_v3,
        }
    return FEATURE_FUNCTIONS


def load_model(model_path: str):
    """Load a trained model from disk (pickle is safe for our own locally-generated models)."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def prepare_data(
    feature_version: str,
    data_path: str = 'data/raw/sample_tickers.xlsx',
    verbose: bool = True,
) -> pd.DataFrame:
    """Load raw data, calculate returns, and engineer features (once for all tickers)."""
    from src.data import load_data, calculate_returns

    df = load_data(data_path)
    df = calculate_returns(df, verbose=verbose)
    df = _get_feature_functions()[feature_version](df)
    return df


def predict_next_period(
    ticker: str,
    date: str,
    model_path: str,
    feature_version: str,
    target: str = 'overnight_return',
    data_path: str = 'data/raw/sample_tickers.xlsx',
    *,
    model=None,
    prepared_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Predict next period return for a ticker.

    For batch predictions pass *model* and *prepared_df* to avoid
    reloading data and the model for every ticker.
    """
    if model is None:
        model = load_model(model_path)

    if prepared_df is not None:
        df = prepared_df
    else:
        df = prepare_data(feature_version, data_path)

    # Filter to date and ticker
    date_pd = pd.to_datetime(date)
    ticker_df = df[(df['Ticker'] == ticker) & (df['DATE'] <= date_pd)]

    if len(ticker_df) == 0:
        raise ValueError(f"No data found for {ticker} up to {date}")

    # Get last row features
    last_row = ticker_df.iloc[-1]

    # Reject if requested date is too far beyond available data
    last_data_date = last_row['DATE']
    days_gap = (date_pd - last_data_date).days
    if days_gap > 7:
        raise ValueError(
            f"Requested date {date} is {days_gap} days beyond last available data "
            f"({last_data_date.date()}). Max gap is 7 days."
        )

    exclude_cols = ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST', 'overnight_return', 'intraday_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = pd.DataFrame([last_row[feature_cols].values], columns=feature_cols)

    # Check for NaN
    if X.isna().any().any():
        raise ValueError(f"Features contain NaN for {ticker} on {date}")

    # Predict
    prediction = model.predict(X)[0]

    return {
        'ticker': ticker,
        'date': date,
        'target': target,
        'predicted_return': prediction,
        'last_data_date': last_row['DATE'].strftime('%Y-%m-%d'),
    }
