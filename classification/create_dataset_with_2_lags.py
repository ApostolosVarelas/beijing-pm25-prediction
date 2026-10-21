"""
Create the classification datasets with 2 days of lagged features.

Aggregates the hourly data into daily means, labels each day with
whether PM2.5 went up (1) or down (-1) compared to the previous day,
adds lags of 1-2 days for every variable and splits the result into:
    - train_data.csv: 2010-2013
    - validate_data.csv: January-June 2014
    - test_data.csv: July-December 2014
"""

from typing import Tuple

import pandas as pd

LAG_COLS = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']


def _load_daily_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load the hourly dataset and aggregate it into daily means.

    The wind direction (cbwd) is encoded as category codes before
    averaging.

    Args:
        file_path (str): Path to the original hourly CSV file.

    Returns:
        pd.DataFrame: One row per day with the mean of each variable.
    """
    data = pd.read_csv(file_path)

    data['cbwd'] = data['cbwd'].astype('category').cat.codes

    data_daily = (
        data
        .groupby(['year', 'month', 'day'], as_index=False)
        .agg({
            'pm2.5': 'mean',
            'DEWP': 'mean',
            'TEMP': 'mean',
            'PRES': 'mean',
            'cbwd': 'mean',
            'Iws': 'mean',
            'Is': 'mean',
            'Ir': 'mean'
        })
    )
    return data_daily


def _add_change_label(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Label each day with the direction of the PM2.5 change and drop the
    days without a change.

    Args:
        data (pd.DataFrame): Daily dataset.

    Returns:
        pd.DataFrame: Dataset with a 'pm2.5_change' column of 1 (up) or
            -1 (down).
    """
    data['pm2.5_change'] = data['pm2.5'].diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    data = data[data['pm2.5_change'] != 0]
    return data


def _split_sets(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation and test sets by date.

    Args:
        data (pd.DataFrame): Dataset with year, month and day columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train
            (2010-2013), validation (first half of 2014) and test
            (second half of 2014) sets.
    """
    train_data = data[data['year'] < 2014]

    validate_data = data[
        (data['year'] == 2014) &
        (
            (data['month'] < 7)
            | ((data['month'] == 6) & (data['day'] <= 30))
        )
    ]

    test_data = data[
        (data['year'] == 2014) &
        (
            (data['month'] > 6)
            | ((data['month'] == 7) & (data['day'] >= 1))
        )
    ]
    return train_data, validate_data, test_data


def main() -> None:
    """
    Build the daily datasets with 2 lags and save them as CSV files.

    Returns:
        None
    """
    file_path = 'PRSA_data_2010.1.1-2014.12.31.csv'
    data = _load_daily_data(file_path)

    data = _add_change_label(data)

    for col in LAG_COLS:
        for i in range(1, 3):
            data[f'{col}_lag{i}'] = data[col].shift(i)

    lag_columns = [f'{col}_lag{i}' for col in LAG_COLS for i in range(1, 3)]
    data.dropna(subset=lag_columns, inplace=True)

    train_data, validate_data, test_data = _split_sets(data)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validate_data)}")
    print(f"Test size: {len(test_data)}")

    train_data.to_csv('train_data.csv', index=False)
    validate_data.to_csv('validate_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)


if __name__ == "__main__":
    main()
