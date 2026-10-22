# Beijing PM2.5 Prediction

Machine learning models that predict **PM2.5 air pollution** (particles
smaller than 2.5 µm) in Beijing from weather data.

The project has two parts:

1. **Regression:** predict the hourly PM2.5 concentration. 13 models are
   compared across 6 feature sets.
2. **Classification:** predict whether the daily PM2.5 level will **rise or
   fall** compared to the previous day. 6 classifiers are compared.

Exact regression turned out to be hard, with a best R² of 0.65. Exploring
the data showed much clearer relationships at the daily level, so the
problem was reframed as classification. There the best model reaches a
**weighted F1-score of 0.86**.

## Dataset

[Beijing PM2.5 Data](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
from the UCI Machine Learning Repository. It has hourly measurements from
2010 to 2014 (43,824 rows).

| Column | Description |
|--------|-------------|
| `year`, `month`, `day`, `hour` | Time of the measurement |
| `pm2.5` | PM2.5 concentration (µg/m³), **the target** |
| `DEWP` | Dew point (°C) |
| `TEMP` | Temperature (°C) |
| `PRES` | Pressure (hPa) |
| `cbwd` | Combined wind direction |
| `Iws` | Cumulated wind speed (m/s) |
| `Is` | Cumulated hours of snow |
| `Ir` | Cumulated hours of rain |

Download `PRSA_data_2010.1.1-2014.12.31.csv` and place it in the
repository root.

## Project structure

```
beijing-pm25-prediction/
├── regression/
│   ├── feature_engineering.py           # Hourly features + train/val/test split
│   └── train_regression_models.py       # Tunes and evaluates 13 regressors
├── classification/
│   ├── create_dataset_with_2_lags.py    # Daily up/down dataset, 2-day lags
│   ├── create_dataset_with_10_lags.py   # Daily up/down dataset, 10-day lags
│   └── train_classifiers.py             # Tunes and evaluates 6 classifiers
├── analysis/
│   └── plot_pm25_vs_iws.py              # PM2.5 vs wind speed plots
└── experiments/                         # Approaches tried and dropped
    ├── add_time_features.py
    ├── create_dataset_with_mean_values.py
    └── find_best_lag_depth.py
```

## Getting started

```bash
git clone https://github.com/ApostolosVarelas/beijing-pm25-prediction.git
cd beijing-pm25-prediction
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

All scripts read and write files in the **current directory**, so run
them from the repository root.

### Regression

```bash
python regression/feature_engineering.py
```

This creates `train_data.csv`, `validation_data.csv` and `test_data.csv`.
The training script expects each feature set in its own subfolder of
`ML_data_crops/`:

```
ML_data_crops/
└── <feature_set_name>/
    ├── train_data.csv
    ├── validation_data.csv
    └── test_data.csv
```

```bash
python regression/train_regression_models.py
```

For every feature set and model, the results (MSE, MAE, RMSE and R² on the
train, validation and test sets) and 5 error analysis plots are saved in
`ML_all_models/<feature_set>/<model>/`.

### Classification

```bash
python classification/create_dataset_with_2_lags.py   # or _10_lags
python classification/train_classifiers.py
```

This saves the classification reports to `classification_results.txt`,
together with confusion matrices, feature importances, ROC curves and an
F1-score comparison plot for each classifier.

### Data analysis

```bash
python analysis/plot_pm25_vs_iws.py
```

This plots PM2.5 against wind speed at monthly, daily and hourly
resolution, in `output_plots/`.

## Method

### Regression (hourly)

Data is split by time: **train** before 2013, **validation** January to
August 2013, **test** from September 2013. Missing values are filled with
backward/forward fill and features are standardized.

13 models are tuned with `GridSearchCV`: Linear Regression, Ridge, Lasso,
ElasticNet, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVR,
k-NN, Decision Tree, Gaussian Process and MLP.

Six feature sets were compared:

1. **Base features:** month, day, hour and the weather variables
2. **+ Cyclical time encoding:** sin/cos of hour, day and month
3. **+ Lags of 1-2 hours** for all variables
4. **+ Lags of 1-4 hours** for all variables
5. **Cyclical encoding + 6 h and 24 h rolling means and deviations**
6. **Cyclical encoding + 10-hour lags** of the most important variables

`feature_engineering.py` shows the lag 1-4 version. It was adjusted to
produce the other feature sets.

### Classification (daily)

Plotting the data showed **no clear relationship** between PM2.5 and the
weather variables at monthly or hourly resolution. At **daily** resolution
there was a strong correlation with dew point (`DEWP`), wind direction
(`cbwd`) and wind speed (`Iws`).

So the hourly data was averaged per day, and each day was labeled **1**
(PM2.5 increased) or **-1** (PM2.5 decreased) compared to the previous
day. The splits are 2010-2013 for training, the first half of 2014 for
validation and the second half of 2014 for testing.

## Results

### Regression: best model per feature set

| Feature set | Best model | MSE | MAE | RMSE | R² |
|-------------|------------|-----|-----|------|----|
| 1. Base features | Extra Trees | 3900.20 | 42.60 | 62.45 | 0.50 |
| 2. + Cyclical time encoding | MLP | 3704.57 | 42.54 | 60.87 | 0.52 |
| 3-4. + Lags of 1-2 / 1-4 hours | MLP | 3582.15 | 41.10 | 59.85 | 0.54 |
| 5. Rolling means and deviations | MLP | 3108.31 | 37.59 | 55.75 | 0.60 |
| 6. **10-hour lags** | **MLP** | **3087.18** | **40.87** | **55.56** | **0.65** |

The MLP was the best model on almost every feature set. Gradient Boosting,
Extra Trees and Random Forest followed closely. Adding temporal context
(lags and rolling statistics) improved R² from 0.50 to 0.65. Large PM2.5
spikes, however, remained consistently underestimated.

### Classification: weighted F1-score

| Model | 2-day lags | 10-day lags |
|-------|-----------|-------------|
| Random Forest | 0.8094 | 0.7946 |
| Gradient Boosting | 0.7900 | 0.8005 |
| **Logistic Regression** | **0.8457** | **0.8575** |
| Decision Tree | 0.7869 | 0.6795 |
| K-Nearest Neighbors | 0.7900 | 0.6973 |
| SVM | 0.8337 | 0.8462 |

**Logistic Regression** performed best with both feature sets, reaching an
F1-score of 0.86 and a ROC AUC of 0.92. The simpler linear models (Logistic
Regression, SVM) generalized better than the tree-based ones. Decision
Tree and k-NN lost a lot of accuracy with the longer 10-day history.

## Experiments

The `experiments/` folder contains approaches that were tested and
dropped because they did not improve the results:

- **`add_time_features.py`** adds cyclical time features to existing
  CSV files.
- **`create_dataset_with_mean_values.py`** builds a classification dataset
  with scaled `DEWP`, `cbwd` and `Iws` features.
- **`find_best_lag_depth.py`** picks the best lag (1-24 hours) per feature
  with 5-fold cross-validation. It reads from a
  `final_feature_engineering/` folder of lagged datasets.

## Built with

pandas, NumPy, scikit-learn, matplotlib and seaborn.
