"""
Find the optimal lag depth for each feature with cross-validation.

For each feature, lag depths from 1 to max_lag are tested separately
with a RandomForestRegressor, using the RMSE from 5-fold
cross-validation. The lag depth with the lowest mean RMSE is selected
as the optimal lag for that feature. A model is then trained on the
selected lagged features and evaluated on the validation and test
sets.

This approach was tested but was not useful enough.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score


def _find_optimal_lag_depth_cv(
    data: pd.DataFrame,
    target: str,
    features: List[str],
    max_lag: int
) -> Dict[str, Tuple[int, float]]:
    """
    Determine the optimal lag depth for each feature using
    cross-validation.

    Args:
        data (pd.DataFrame): Dataset with lagged features.
        target (str): Target column name.
        features (List[str]): Feature names (without the lag suffix).
        max_lag (int): Maximum lag depth to evaluate.

    Returns:
        Dict[str, Tuple[int, float]]: Mapping of each feature to its
            optimal (lag, mean RMSE).
    """
    results = {}
    X = data.drop(columns=[target])
    y = data[target]

    for feature in features:
        rmse_scores = []
        for lag in range(1, max_lag + 1):
            lagged_feature = f'{feature}_lag{lag}'
            if lagged_feature in data.columns:
                X_subset = X[[lagged_feature]]

                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                model = RandomForestRegressor(random_state=42)

                scorer = make_scorer(mean_squared_error, squared=False)

                cv_scores = cross_val_score(
                    model, X_subset, y, cv=kf, scoring=scorer
                )
                mean_rmse = np.mean(cv_scores)
                rmse_scores.append((lag, mean_rmse))

        optimal_lag = min(rmse_scores, key=lambda x: x[1])
        results[feature] = optimal_lag

    return results


def _select_rows_without_nan(
    data: pd.DataFrame,
    features: List[str],
    target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select the features and target, keeping only the rows where no
    feature is missing.

    Args:
        data (pd.DataFrame): Dataset to select from.
        features (List[str]): Feature column names.
        target (str): Target column name.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The features and the target.
    """
    X = data[features]
    y = data[target]

    X = X.dropna()
    y = y.loc[X.index]
    return X, y


def main() -> None:
    """
    Find the optimal lags, then train and evaluate a model on them.

    Returns:
        None
    """
    train_data_path = 'final_feature_engineering/train_data_with_lags.csv'
    validation_data_path = (
        'final_feature_engineering/validation_data_with_lags.csv'
    )
    test_data_path = 'final_feature_engineering/test_data_with_lags.csv'

    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)
    test_data = pd.read_csv(test_data_path)

    target_column = 'pm2.5'
    original_features = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
    max_lag_depth = 24

    optimal_lags = _find_optimal_lag_depth_cv(
        train_data, target_column, original_features, max_lag_depth
    )

    print("Optimal lag depths for each feature:")
    print(optimal_lags)

    selected_lag_features = [
        f"{feature}_lag{optimal_lags[feature][0]}"
        for feature in optimal_lags
    ]
    X_train, y_train = _select_rows_without_nan(
        train_data, selected_lag_features, target_column
    )
    X_val, y_val = _select_rows_without_nan(
        validation_data, selected_lag_features, target_column
    )

    model = RandomForestRegressor(
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        n_estimators=100
    )
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))

    print(f"Validation RMSE: {val_rmse}")

    X_test, y_test = _select_rows_without_nan(
        test_data, selected_lag_features, target_column
    )

    test_predictions = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    print(f"Test RMSE: {test_rmse}")


if __name__ == "__main__":
    main()
