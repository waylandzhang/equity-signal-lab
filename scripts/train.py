#!/usr/bin/env python3
"""Train prediction models for stock returns."""

import argparse
import os
import sys
import random
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_data, calculate_returns, check_stationarity
from src.features import engineer_features_v1, engineer_features_v2, engineer_features_v3
from src.models import train_ridge, train_elastic_net, train_lightgbm
from src.cv import time_series_cv_split
from src.evaluation import (
    calculate_metrics, compare_models, plot_predictions_vs_actuals,
    plot_residuals_over_time, calculate_sharpe_ratio
)

SEED = 42

FEATURE_FUNCTIONS = {
    'v1': engineer_features_v1,
    'v2': engineer_features_v2,
    'v3': engineer_features_v3
}

MODEL_FUNCTIONS = {
    'ridge': train_ridge,
    'elasticnet': train_elastic_net,
    'lgbm': train_lightgbm
}


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility across the full pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_data(feature_version: str, target: str, data_path: str):
    """Load data, compute returns and features. Returns full DataFrame."""
    print(f"\n{'='*60}")
    print(f"Loading data and engineering {feature_version} features...")
    print(f"{'='*60}")

    df = load_data(data_path)
    df = calculate_returns(df)

    # Check stationarity on first ticker
    first_ticker = df['Ticker'].iloc[0]
    ticker_data = df[df['Ticker'] == first_ticker][target].dropna()
    check_stationarity(ticker_data, f'{first_ticker} {target}')

    # Engineer features
    df = FEATURE_FUNCTIONS[feature_version](df)

    # Identify feature columns
    exclude_cols = ['Ticker', 'DATE', 'PX_OPEN', 'PX_LAST',
                    'overnight_return', 'intraday_return']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Drop rows where features or target are NaN
    required = feature_cols + [target, 'DATE']
    df_clean = df.dropna(subset=required).reset_index(drop=True)

    print(f"Features: {len(feature_cols)} columns, {len(df_clean)} samples")
    print(f"Target: {target}")

    return df_clean, feature_cols, target


def train_and_evaluate(model_name: str, df, feature_cols, target,
                       min_train_days: int = 21, max_train_days: int = 756):
    """Train model using proper time-series cross-validation."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with time-series CV "
          f"(min={min_train_days}d, max={max_train_days}d)...")
    print(f"{'='*60}")

    model_fn = MODEL_FUNCTIONS[model_name]

    # Collect out-of-sample predictions across all CV folds
    all_preds = []
    all_actuals = []
    all_dates = []
    fold_r2 = []

    splits = list(time_series_cv_split(df, min_train_days, max_train_days))
    n_folds = len(splits)
    print(f"  CV folds: {n_folds}")

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, target]

        result = model_fn(X_train, y_train)
        preds = result['model'].predict(X_test)

        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        all_dates.extend(df.loc[test_idx, 'DATE'].values)

        if (i + 1) % 100 == 0:
            print(f"  Fold {i+1}/{n_folds} done")

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_dates = pd.to_datetime(all_dates)

    # Aggregate out-of-sample metrics (with dates for cross-sectional IC)
    cv_metrics = calculate_metrics(all_actuals, all_preds, dates=all_dates.values)

    # Portfolio-level Sharpe: aggregate to daily mean, then annualize
    strategy_returns = np.sign(all_preds) * all_actuals
    daily_returns = pd.Series(strategy_returns, index=all_dates).groupby(level=0).mean()
    cv_metrics['sharpe'] = calculate_sharpe_ratio(daily_returns.values)

    print(f"\nOut-of-sample CV Metrics ({n_folds} folds):")
    for k, v in cv_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Train final model on ALL data (for saving / predictions)
    final_result = model_fn(df[feature_cols], df[target])

    return (final_result['model'], cv_metrics,
            all_actuals, all_preds, all_dates)


def main():
    parser = argparse.ArgumentParser(
        description='Train returns prediction models')
    parser.add_argument('--models', type=str, default='ridge',
                        help='Comma-separated: ridge,elasticnet,lgbm')
    parser.add_argument('--features', type=str, default='v1',
                        help='Comma-separated: v1,v2,v3')
    parser.add_argument('--data', type=str, default='data/raw/sample_tickers.xlsx',
                        help='Path to input Excel data file')
    parser.add_argument('--target', type=str, default='overnight_return',
                        choices=['overnight_return', 'intraday_return'])
    parser.add_argument('--min-train', type=int, default=21,
                        help='Minimum training window (trading days)')
    parser.add_argument('--max-train', type=int, default=756,
                        help='Maximum training window (trading days)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory for models, figures, and comparison CSV')

    args = parser.parse_args()

    set_seed(SEED)

    models = args.models.split(',')
    features = args.features.split(',')

    output_dir = args.output_dir
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    all_results = {}

    for feat_ver in features:
        df_clean, feature_cols, target = prepare_data(feat_ver, args.target, args.data)

        for model_name in models:
            key = f"{model_name}_{feat_ver}"

            model, metrics, y_true, y_pred, dates = train_and_evaluate(
                model_name, df_clean, feature_cols, target,
                args.min_train, args.max_train
            )
            all_results[key] = metrics

            # Save model (pickle for sklearn models — locally trained, trusted)
            model_path = f'{output_dir}/models/{key}_{args.target}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"\nModel saved: {model_path}")

            # Save scatter plot
            plot_path = f'{output_dir}/figures/{key}_{args.target}_predictions.png'
            plot_predictions_vs_actuals(
                y_true, y_pred,
                title=f"{key} - {args.target} (CV out-of-sample)",
                save_path=plot_path)
            print(f"Plot saved: {plot_path}")

            # Save residuals-over-time plot
            resid_path = f'{output_dir}/figures/{key}_{args.target}_residuals.png'
            plot_residuals_over_time(
                dates, y_pred - y_true,
                title=f"{key} - Residuals over time",
                save_path=resid_path)
            print(f"Residuals plot saved: {resid_path}")

    # Compare all models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON (out-of-sample CV)")
    print(f"{'='*60}")
    comparison = compare_models(all_results)
    print(comparison.to_string())

    comparison.to_csv(f'{output_dir}/model_comparison_{args.target}.csv')
    print(f"\nComparison saved: {output_dir}/model_comparison_{args.target}.csv")


if __name__ == '__main__':
    main()
