"""
Create the hourly regression datasets.

Loads the air quality data, fills missing values with backward-fill and
then forward-fill, and splits the data by date into:
    - train: before 2013-01-01
    - validation: 2013-01-01 to 2013-08-31
    - test: from 2013-09-01 onward

Feature engineering:
    - Encodes hour, month and day cyclically with sin and cos.
    - Encodes the wind direction (cbwd) into numerical categories.
    - Drops unnecessary columns (No, year).
    - Adds lags of 1 to 4 hours for DEWP, TEMP, PRES, Iws, Is and Ir.

The lags were changed between experiments to create the different
feature sets. The resulting train_data.csv, validation_data.csv and
test_data.csv are saved in the current directory.
"""

from typing import List

import numpy as np
import pandas as pd


def _load_data(
    file_path: str
) -> pd.DataFrame:
    """
    Load the original hourly dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_path)
    return df


def _handle_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing values with backward-fill and then forward-fill.

    Args:
        df (pd.DataFrame): Dataset with missing values.

    Returns:
        pd.DataFrame: Dataset without missing values.
    """
    return df.bfill().ffill()


def _feature_engineering(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add cyclical time features, encode the wind direction and drop
    unnecessary columns.

    Args:
        df (pd.DataFrame): Dataset with hour, month, day, cbwd, No and
            year columns.

    Returns:
        pd.DataFrame: Copy of the dataset with the new features.
    """
    df = df.copy()

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    wind_directions = {'NW': 0, 'NE': 1, 'SE': 2, 'SW': 3}
    df['cbwd'] = df['cbwd'].map(wind_directions)

    df.drop(['No', 'year'], axis=1, inplace=True)

    return df


def _create_lagged_features(
    df: pd.DataFrame,
    features: List[str]
) -> pd.DataFrame:
    """
    Add lags of 1 to 4 time steps for each feature and drop the rows
    with missing values.

    Args:
        df (pd.DataFrame): Dataset to add the lags to.
        features (List[str]): Names of the columns to lag.

    Returns:
        pd.DataFrame: Copy of the dataset with the lagged features.
    """
    df = df.copy()
    for feature in features:
        df.loc[:, f'{feature}_lag_1'] = df[feature].shift(1)
        df.loc[:, f'{feature}_lag_2'] = df[feature].shift(2)
        df.loc[:, f'{feature}_lag_3'] = df[feature].shift(3)
        df.loc[:, f'{feature}_lag_4'] = df[feature].shift(4)

    df.dropna(inplace=True)
    return df


def main() -> None:
    """
    Build the train, validation and test datasets and save them as CSV
    files.

    Returns:
        None
    """
    file_path = "PRSA_data_2010.1.1-2014.12.31.csv"

    df = _load_data(file_path)

    df = _handle_missing_values(df)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    train_df = df[df['date'] < '2013-01-01']
    validation_df = df[
        (df['date'] >= '2013-01-01') & (df['date'] < '2013-09-01')
    ]
    test_df = df[df['date'] >= '2013-09-01']

    train_df = _feature_engineering(train_df)
    validation_df = _feature_engineering(validation_df)
    test_df = _feature_engineering(test_df)

    features_to_lag = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']

    train_df = _create_lagged_features(train_df, features_to_lag)
    validation_df = _create_lagged_features(validation_df, features_to_lag)
    test_df = _create_lagged_features(test_df, features_to_lag)

    train_df.drop(['date'], axis=1, inplace=True)
    validation_df.drop(['date'], axis=1, inplace=True)
    test_df.drop(['date'], axis=1, inplace=True)

    train_df.to_csv("train_data.csv", index=False)
    validation_df.to_csv("validation_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    print("DONE")


if __name__ == "__main__":
    main()
