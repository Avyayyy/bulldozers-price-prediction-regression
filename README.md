# Bulldozer Price Prediction 🚜

Predicting the sale price of used bulldozers using machine learning, based on the [Kaggle Bluebook for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview) competition.

## Problem Statement

The goal is to predict the sale price of bulldozers at auction, given data about the bulldozer's usage, equipment type, and configuration. This is a **regression** problem, and the competition's evaluation metric is **RMSLE (Root Mean Squared Log Error)** between the actual and predicted auction prices.

## Dataset

The data comes from the Kaggle competition page linked above and contains **412,698 sales records** of bulldozers (split into 401,125 training rows and 11,573 validation rows from 2012), including:
- Sale date
- Machine details (model, product class, etc.)
- Usage (e.g. machine hours)
- Configuration (e.g. blade type, tyre size)

> **Note:** The raw data files are not included in this repository (see `.gitignore`) due to their size and Kaggle's terms of use. To reproduce this project, download the data directly from the [competition page](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data) (requires a free Kaggle account) and place it in a local `data/` folder.

## Approach

1. **Exploratory Data Analysis (EDA)** — inspected data types, missing values, and the distribution/trend of `SalePrice` over time.
2. **Date feature engineering** — parsed the `saledate` column and broke it into `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, and `saleDayOfYear`, since this is a time-series problem.
3. **Sorted chronologically** — sorted the dataset by `saledate` since order matters for time-based validation.
4. **Handling missing data:**
   - Numeric columns: filled with the column median, plus an added `_is_missing` binary flag column so the model can still learn from "missingness" as a signal.
   - Categorical columns: converted to pandas `category` dtype, then encoded as numeric category codes (+1, since pandas encodes unknown categories as -1), also with an `_is_missing` flag.
5. **Train/validation split** — used a **time-based split** (not random) to mimic the competition setup: rows from 2012 became the validation set, everything before became the training set.
6. **Modelling** — `RandomForestRegressor` (scikit-learn).
7. **Baseline & fast iteration** — first trained on a subset via `max_samples` to iterate quickly before scaling to the full dataset.
8. **Hyperparameter tuning** — used `RandomizedSearchCV` (100 iterations, 5-fold CV) across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
9. **Final model** — retrained a `RandomForestRegressor` on the full training set using the best hyperparameters found:
   - `n_estimators = 40`
   - `min_samples_leaf = 1`
   - `min_samples_split = 10`
   - `max_features = 0.5`
   - `max_samples = None`
10. **Evaluation** — a custom `rmsle()` function plus MAE and R², computed separately on training and validation sets.
11. **Predictions on the Kaggle test set** — applied the same preprocessing pipeline to `Test.csv`, aligned its columns with the training set, and exported predictions in Kaggle's required submission format.
12. **Feature importance** — plotted the top 20 most important features from the final Random Forest model to interpret what drives `SalePrice`.

## Results

Final tuned Random Forest performance:

| Metric | Training | Validation |
|--------|----------|------------|
| MAE | 2,608.77 | 5,935.52 |
| RMSLE | 0.1301 | 0.2459 |
| R² | 0.9673 | 0.8824 |

The tuned model explains about **88% of the variance** in bulldozer sale prices on unseen (2012) data, with an RMSLE of ~0.246 — meaning predictions are typically within roughly 25% of the true sale price.

**Feature importance:** `YearMade` was by far the strongest predictor, followed by `ProductSize`. A second tier of features — including `Enclosure`, `Differential_Type`, and the product-family descriptor fields (`fiSecondaryDesc`, `fiBaseModel`, `fiModelDesc`) — contributed smaller but meaningful signal. Full ranked chart is in the notebook.

## Project Structure

```
.
├── README.md
├── .gitignore
├── end-to-end-bulldozer-price-regression.ipynb   # main notebook
└── data/                                          # (not tracked in git — download separately)
    ├── Train.csv
    ├── Valid.csv
    └── Test.csv
```

## Setup & Installation

This project uses **Miniconda** for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/end-to-end-bulldozer-price-regression.git
cd end-to-end-bulldozer-price-regression

# Create and activate a conda environment
conda create -n bulldozer-regression python=3.10
conda activate bulldozer-regression

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

<!-- TODO: generate a requirements.txt with: conda list -e > requirements.txt (or pip freeze > requirements.txt) -->

## Tools & Libraries

- Python
- pandas, NumPy
- scikit-learn (Random Forest Regressor)
- Matplotlib / Seaborn (visualization)
- Jupyter Notebook

## Acknowledgements

- [Kaggle Bluebook for Bulldozers Competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview)
