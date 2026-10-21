"""
Add cyclical time features to already created CSV files.

Reads train_data.csv, test_data.csv and validation_data.csv, encodes
month, day and hour with sin and cos, and saves the results as
*_with_time_features.csv.
"""

import numpy as np
import pandas as pd


def _add_time_features(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Add sin and cos encodings of the month, day and hour.

    Args:
        data (pd.DataFrame): Dataset with month, day and hour columns.

    Returns:
        pd.DataFrame: The same dataset with the new columns.
    """
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    return data


def main() -> None:
    """
    Add the time features to the three datasets and save them.

    Returns:
        None
    """
    # Load datasets
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    validation_data = pd.read_csv('validation_data.csv')

    # Apply time engineering
    train_data = _add_time_features(train_data)
    test_data = _add_time_features(test_data)
    validation_data = _add_time_features(validation_data)

    # Save the modified datasets
    train_data.to_csv('train_data_with_time_features.csv', index=False)
    test_data.to_csv('test_data_with_time_features.csv', index=False)
    validation_data.to_csv(
        'validation_data_with_time_features.csv', index=False
    )


if __name__ == "__main__":
    main()
