#!/usr/bin/env python3
"""Predict next period returns using trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import predict_next_period, load_model, prepare_data
from src.data import load_data

def main():
    parser = argparse.ArgumentParser(description='Predict stock returns')
    parser.add_argument('--ticker', type=str, help='Ticker symbol (e.g., "NVDA")')
    parser.add_argument('--date', type=str, required=True,
                       help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='ridge',
                       choices=['ridge', 'elasticnet', 'lgbm'],
                       help='Model to use')
    parser.add_argument('--features', type=str, default='v1',
                       choices=['v1', 'v2', 'v3'],
                       help='Feature version')
    parser.add_argument('--target', type=str, default='overnight_return',
                       choices=['overnight_return', 'intraday_return'],
                       help='Target to predict')
    parser.add_argument('--data', type=str, default='data/raw/sample_tickers.xlsx',
                       help='Path to input Excel data file')

    args = parser.parse_args()

    # Model path
    model_path = f'results/models/{args.model}_{args.features}_{args.target}.pkl'

    # Load model and prepare data once
    model = load_model(model_path)
    df = prepare_data(args.features, args.data, verbose=True)

    # Determine tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = df['Ticker'].unique()

    print(f"\n{'='*60}")
    print(f"Predictions for {args.date}")
    print(f"Model: {args.model}_{args.features}, Target: {args.target}")
    print(f"{'='*60}\n")

    for ticker in tickers:
        try:
            result = predict_next_period(
                ticker=ticker,
                date=args.date,
                model_path=model_path,
                feature_version=args.features,
                target=args.target,
                data_path=args.data,
                model=model,
                prepared_df=df,
            )

            print(f"{ticker}:")
            print(f"  Predicted {args.target}: {result['predicted_return']:.4f} ({result['predicted_return']*100:.2f}%)")
            print(f"  Last data date: {result['last_data_date']}\n")

        except Exception as e:
            print(f"{ticker}: ERROR - {e}\n")

if __name__ == '__main__':
    main()
