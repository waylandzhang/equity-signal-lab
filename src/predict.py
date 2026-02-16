import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any

def predict_next_period(
    ticker: str,
    date: str,
    model_path: str,
    feature_version: str,
    target: str = 'overnight_return',
    data_path: str = 'data/raw/sample_tickers.xlsx'
) -> Dict[str, Any]:
    """
    Predict next period return for a ticker.

    Uses pickle for model loading (standard ML practice for locally-generated trusted models).
    """
    from src.data import load_data, calculate_returns
    from src.features import engineer_features_v1, engineer_features_v2, engineer_features_v3

    FEATURE_FUNCTIONS = {
        'v1': engineer_features_v1,
        'v2': engineer_features_v2,
        'v3': engineer_features_v3
    }

    # Load model (pickle is safe for our own trained models)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data up to date
    df = load_data(data_path)
    df = calculate_returns(df)
    df = FEATURE_FUNCTIONS[feature_version](df)

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
        'last_data_date': last_row['DATE'].strftime('%Y-%m-%d')
    }
