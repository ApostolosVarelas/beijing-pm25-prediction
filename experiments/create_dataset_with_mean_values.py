"""
Create daily classification datasets with scaled features and lags.

Creates train_data.csv, validation_data.csv and test_data.csv from the
original dataset. On top of the daily means, they contain:
    - lags of 1-2 days for TEMP, PRES, Is and Ir
    - DEWP, cbwd and Iws scaled around their mean, with lags of 0-2 days

This approach was tested but was not useful enough.
"""

from typing import Tuple

import pandas as pd

LAG_COLS = ['TEMP', 'PRES', 'Is', 'Ir']


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


def _add_scaled_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add DEWP, cbwd and Iws scaled around their mean, with lags of 0-2
    days.

    Each variable is scaled as ((x - mean) / (max - min)) * 20 and then
    centered on zero.

    Args:
        data (pd.DataFrame): Daily dataset.

    Returns:
        pd.DataFrame: The dataset with the scaled features and lags.
    """
    min_dewp = data['DEWP'].min()
    max_dewp = data['DEWP'].max()
    mean_dewp = data['DEWP'].mean()

    data['DEWP_scaled'] = (
        ((data['DEWP'] - mean_dewp) / (max_dewp - min_dewp)) * 20
    )

    data['DEWP_scaled'] = data['DEWP_scaled'] - data['DEWP_scaled'].mean()

    data['DEWP_scaled_lag0'] = data['DEWP_scaled']
    data['DEWP_scaled_lag1'] = data['DEWP_scaled'].shift(1)
    data['DEWP_scaled_lag2'] = data['DEWP_scaled'].shift(2)

    min_cbwd = data['cbwd'].min()
    max_cbwd = data['cbwd'].max()
    mean_cbwd = data['cbwd'].mean()

    min_iws = data['Iws'].min()
    max_iws = data['Iws'].max()
    mean_iws = data['Iws'].mean()

    data['cbwd_scaled'] = (
        ((data['cbwd'] - mean_cbwd) / (max_cbwd - min_cbwd)) * 20
    )
    data['cbwd_scaled'] = data['cbwd_scaled'] - data['cbwd_scaled'].mean()

    data['Iws_scaled'] = ((data['Iws'] - mean_iws) / (max_iws - min_iws)) * 20
    data['Iws_scaled'] = data['Iws_scaled'] - data['Iws_scaled'].mean()

    data['cbwd_scaled_lag0'] = data['cbwd_scaled']
    data['cbwd_scaled_lag1'] = data['cbwd_scaled'].shift(1)
    data['cbwd_scaled_lag2'] = data['cbwd_scaled'].shift(2)

    data['Iws_scaled_lag0'] = data['Iws_scaled']
    data['Iws_scaled_lag1'] = data['Iws_scaled'].shift(1)
    data['Iws_scaled_lag2'] = data['Iws_scaled'].shift(2)

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
    Build the datasets and save them as CSV files.

    Returns:
        None
    """
    file_path = 'PRSA_data_2010.1.1-2014.12.31.csv'
    data = _load_daily_data(file_path)

    data['pm2.5_change'] = data['pm2.5'].diff().apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )

    data = data[data['pm2.5_change'] != 0]

    for col in LAG_COLS:
        for i in range(1, 3):
            data[f'{col}_lag{i}'] = data[col].shift(i)

    data = _add_scaled_features(data)

    lag_columns = [f'{col}_lag{i}' for col in LAG_COLS for i in range(1, 3)]
    lag_columns.extend(
        ['DEWP_scaled_lag0', 'DEWP_scaled_lag1', 'DEWP_scaled_lag2']
    )
    lag_columns.extend([
        'cbwd_scaled_lag0', 'cbwd_scaled_lag1', 'cbwd_scaled_lag2',
        'Iws_scaled_lag0', 'Iws_scaled_lag1', 'Iws_scaled_lag2'
    ])

    data.dropna(subset=lag_columns, inplace=True)

    train_data, validate_data, test_data = _split_sets(data)

    train_data.drop(columns=['year'], inplace=True)
    validate_data.drop(columns=['year'], inplace=True)
    test_data.drop(columns=['year'], inplace=True)

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validate_data)}")
    print(f"Test size: {len(test_data)}")

    train_data.to_csv('train_data.csv', index=False)
    validate_data.to_csv('validation_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)


if __name__ == "__main__":
    main()
