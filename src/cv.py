import pandas as pd
import numpy as np
from typing import Iterator, Tuple

def time_series_cv_split(
    df: pd.DataFrame,
    min_train_days: int = 21,
    max_train_days: int = 756
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time series cross-validation splits.

    Uses expanding window until max_train_days, then sliding window.
    All tickers are split at the same dates to preserve cross-sectional relationships.

    Args:
        df: DataFrame with 'DATE' column
        min_train_days: Minimum training days (1 month)
        max_train_days: Maximum training days (3 years)

    Yields:
        (train_indices, test_indices)
    """
    dates = sorted(df['DATE'].unique())

    for i in range(min_train_days, len(dates)):
        test_date = dates[i]

        # Expanding window until max_train_days
        if i <= max_train_days:
            train_dates = dates[:i]
        else:
            # Sliding window after hitting max
            train_dates = dates[i - max_train_days:i]

        train_idx = df[df['DATE'].isin(train_dates)].index.values
        test_idx = df[df['DATE'] == test_date].index.values

        yield (train_idx, test_idx)
