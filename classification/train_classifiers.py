"""
Train and evaluate classifiers that predict whether the daily PM2.5
concentration goes up or down compared to the previous day.

Reads train_data.csv, validate_data.csv and test_data.csv (created by one
of the create_dataset_with_*_lags.py scripts), tunes each classifier with
GridSearchCV, and saves classification reports, confusion matrices,
feature importances, ROC curves and an F1-score comparison plot.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

NON_FEATURE_COLUMNS = ['pm2.5_change', 'year', 'month', 'day', 'pm2.5']
TARGET_COLUMN = 'pm2.5_change'


def _split_features_target(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into features and target.

    Args:
        data (pd.DataFrame): Dataset containing the target and the
            non-feature date columns.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The features and the target.
    """
    X = data.drop(columns=NON_FEATURE_COLUMNS)
    y = data[TARGET_COLUMN]
    return X, y


def _get_classifiers() -> Dict[str, Tuple[BaseEstimator, Dict[str, List]]]:
    """
    Build the classifiers and their hyperparameter grids.

    Returns:
        Dict[str, Tuple[BaseEstimator, Dict[str, List]]]: Mapping of
            classifier name to (estimator, parameter grid).
    """
    return {
        'RandomForest': (
            RandomForestClassifier(), {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(), {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10]
            }
        ),
        'LogisticRegression': (
            LogisticRegression(max_iter=1000), {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        ),
        'DecisionTree': (
            DecisionTreeClassifier(), {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'KNeighbors': (
            KNeighborsClassifier(), {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        ),
        'SVM': (
            SVC(probability=True), {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        ),
    }


def _evaluate_on_set(
    name: str,
    set_name: str,
    grid_search: GridSearchCV,
    X: pd.DataFrame,
    y: pd.Series,
    labels: np.ndarray,
    f1_scores: Dict[str, List],
    output_text: List[str]
) -> None:
    """
    Evaluate a tuned classifier on one dataset and save its confusion
    matrix.

    Prints the classification report and the weighted F1-score, adds them
    to the text output and records the F1-score for the comparison plot.

    Args:
        name (str): Name of the classifier.
        set_name (str): Name of the dataset ('Validation' or 'Test').
        grid_search (GridSearchCV): The fitted grid search.
        X (pd.DataFrame): Scaled features of the dataset.
        y (pd.Series): True labels of the dataset.
        labels (np.ndarray): Class labels shown on the confusion matrix.
        f1_scores (Dict[str, List]): Collected F1-scores, updated in
            place.
        output_text (List[str]): Lines of the text report, updated in
            place.

    Returns:
        None
    """
    y_pred = grid_search.predict(X)
    print(f"  {set_name} Report:")
    print(classification_report(y, y_pred))
    output_text.append(f"  {set_name} Report:")
    output_text.append(classification_report(y, y_pred))

    f1_weighted = f1_score(y, y_pred, average='weighted')
    print(f"  Weighted F1-Score ({set_name}): {f1_weighted:.4f}")
    output_text.append(
        f"  Weighted F1-Score ({set_name}): {f1_weighted:.4f}"
    )
    f1_scores['Classifier'].append(name)
    f1_scores['Set'].append(set_name)
    f1_scores['F1-Score'].append(f1_weighted)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title(f'{name} - Confusion Matrix ({set_name})')
    plt.savefig(f'{name}_confusion_matrix_{set_name.lower()}.png')
    plt.close()


def _plot_feature_importances(
    name: str,
    estimator: BaseEstimator,
    features: pd.Index
) -> None:
    """
    Save a bar plot of the feature importances, if the estimator has
    them.

    Args:
        name (str): Name of the classifier.
        estimator (BaseEstimator): The best estimator found by the grid
            search.
        features (pd.Index): Names of the features.

    Returns:
        None
    """
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        plt.title(f'{name} - Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig(f'{name}_feature_importances.png')
        plt.close()


def _plot_roc_curve(
    name: str,
    grid_search: GridSearchCV,
    X_validate: pd.DataFrame,
    y_validate: pd.Series
) -> None:
    """
    Save the ROC curve on the validation set, if the estimator can
    output scores.

    Args:
        name (str): Name of the classifier.
        grid_search (GridSearchCV): The fitted grid search.
        X_validate (pd.DataFrame): Scaled validation features.
        y_validate (pd.Series): Validation labels.

    Returns:
        None
    """
    best_estimator = grid_search.best_estimator_
    if (hasattr(best_estimator, "predict_proba")
            or hasattr(best_estimator, "decision_function")):
        plt.figure(figsize=(8, 6))
        if hasattr(best_estimator, "predict_proba"):
            y_val_prob = grid_search.predict_proba(X_validate)[:, 1]
        else:
            y_val_prob = grid_search.decision_function(X_validate)

        fpr, tpr, _ = roc_curve(
            y_validate, y_val_prob, pos_label=np.unique(y_validate)[-1]
        )
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(f'{name}_roc_curve.png')
        plt.close()


def _plot_f1_comparison(
    f1_scores: Dict[str, List]
) -> None:
    """
    Save a bar plot comparing the weighted F1-scores of all classifiers
    on the validation and test sets.

    Args:
        f1_scores (Dict[str, List]): Collected F1-scores with the keys
            'Classifier', 'Set' and 'F1-Score'.

    Returns:
        None
    """
    print("F1-Score Comparison Plot")
    f1_df = pd.DataFrame(f1_scores)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=f1_df, x='Classifier', y='F1-Score', hue='Set')
    plt.title('F1-Score Comparison Across Classifiers')
    plt.ylabel('Weighted F1-Score')
    plt.xlabel('Classifier')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig('f1_score_comparison.png')
    plt.close()


def main() -> None:
    """
    Load the datasets, scale the features, tune and evaluate every
    classifier, and save all reports and plots.

    Returns:
        None
    """
    train_data = pd.read_csv('train_data.csv')
    train_data = train_data.dropna()
    validate_data = pd.read_csv('validate_data.csv')
    test_data = pd.read_csv('test_data.csv')

    # ------------------ FEATURE & TARGET SELECTION ---------------------
    X_train, y_train = _split_features_target(train_data)
    X_validate, y_validate = _split_features_target(validate_data)
    X_test, y_test = _split_features_target(test_data)

    # ------------------ STANDARDIZATION -------------------------------
    feature_names = X_train.columns

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_validate = pd.DataFrame(
        scaler.transform(X_validate), columns=feature_names
    )
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    labels = np.unique(y_train)

    f1_scores = {'Classifier': [], 'Set': [], 'F1-Score': []}
    # ------------------ GRID SEARCH & EVALUATION -----------------------
    for name, (clf, param_grid) in _get_classifiers().items():
        output_text = []
        output_text.append(f"\nClassifier: {name}")

        print(f"\nClassifier: {name}")
        grid_search = GridSearchCV(
            clf, param_grid, cv=3, scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)

        print("  Best Parameters:", grid_search.best_params_)
        output_text.append(f"  Best Parameters: {grid_search.best_params_}")

        _evaluate_on_set(
            name, 'Validation', grid_search, X_validate, y_validate,
            labels, f1_scores, output_text
        )
        _evaluate_on_set(
            name, 'Test', grid_search, X_test, y_test,
            labels, f1_scores, output_text
        )

        _plot_feature_importances(
            name, grid_search.best_estimator_, X_train.columns
        )
        _plot_roc_curve(name, grid_search, X_validate, y_validate)

        output_file_path = "classification_results.txt"

        with open(output_file_path, "a+") as file:
            file.write("\n".join(output_text))

    _plot_f1_comparison(f1_scores)


if __name__ == "__main__":
    main()
