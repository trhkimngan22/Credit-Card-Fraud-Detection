import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from feature_engine.outliers import Winsorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define random_state
RANDOM_STATE = 42


class OutlierRemover(BaseEstimator):
    """
    Custom sampler wrapping IsolationForest to remove outliers from both X and y.

    Behavior:
        - By implementing `fit_resample(X, y)`, imblearn Pipeline treats this as a Sampler.
        - During training, it removes outliers from both X and y.
        - During testing/prediction, imblearn Pipeline AUTOMATICALLY bypasses samplers,
          so no test data is dropped.
    """
    _estimator_type = "sampler"

    def __init__(self, n_estimators=200, contamination=0.01, random_state=RANDOM_STATE):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Remove outliers and return modified X and y so they remain synchronized.
        """
        self.iso_forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit IsolationForest and predict outliers in one step
        preds = self.iso_forest_.fit_predict(X)
        mask = preds == 1  # 1 = inlier, -1 = outlier

        # Filter X
        if isinstance(X, pd.DataFrame):
            X_filtered = X.loc[mask].reset_index(drop=True)
        else:
            X_filtered = X[mask]

        # Filter y
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_filtered = y.loc[mask].reset_index(drop=True)
        else:
            y_filtered = y[mask]

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"OutlierRemover: removed {n_removed} outliers ({n_removed / len(X) * 100:.2f}%)")

        return X_filtered, y_filtered

    def get_feature_names_out(self, input_features=None):
        return input_features


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def remove_duplicated_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated data."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate rows")
    else:
        logger.info("No duplicate rows found")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split data into training and hold-out test sets.

    Args:
        df: DataFrame with features and 'Class' target column.
        test_size: Proportion of data to hold out for final testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    logger.info(f"Train class distribution:\n{y_train.value_counts(normalize=True).to_string()}")

    return X_train, X_test, y_train, y_test


def create_preprocessing_pipeline(
    model=None,
    contamination: float = 0.01,
    sampling_strategy: str = 'auto',
    k_neighbors: int = 5
) -> Pipeline:
    """
    Create an imblearn Pipeline with preprocessing steps and optionally a model.

    Pipeline steps:
        1. Winsorizer  — cap extreme values (1st & 99th percentile)
        2. StandardScaler — normalize features
        3. OutlierRemover — remove outliers via IsolationForest (custom transformer)
        4. SMOTE — oversample minority class
        5. Model

    Args:
        model: Sklearn estimator to append to the pipeline.
        contamination: Expected proportion of outliers for IsolationForest.
        sampling_strategy: SMOTE sampling_strategy (ratio of minority to majority after resampling).
        k_neighbors: Number of nearest neighbors for SMOTE.

    Returns:
        Imblearn Pipeline

    Usage with cross-validation:
        from sklearn.model_selection import cross_validate, StratifiedKFold

        pipe = create_preprocessing_pipeline(model=LogisticRegression())
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_validate(pipe, X_train, y_train, cv=cv,
                                 scoring=['accuracy', 'f1', 'roc_auc'])
    """
    steps = [
        ("capper", Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
        ("scaler", RobustScaler()),
        ("outlier_remover", OutlierRemover(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE
        )),
        ("resampler", SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=RANDOM_STATE
        )),
    ]

    if model is not None:
        steps.append(("model", model))

    return Pipeline(steps)


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Preprocess data: remove duplicates and split into train/test.

    Args:
        df: Raw DataFrame.
        test_size: Proportion for hold-out test set.

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Remove duplicates
    df = remove_duplicated_data(df)
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)
    return X_train, X_test, y_train, y_test


def save_data(data, file_path: str) -> None:
    """Save data (DataFrame or Series) to CSV file."""
    try:
        logger.info(f"Saving data to {file_path}")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_csv(file_path, index=False)
        else:
            pd.DataFrame(data).to_csv(file_path, index=False)
        logger.info(f"Data saved successfully")
    except Exception as e:
        logger.error(f"Error saving data: {e}")


def main():
    """Main function."""
    df = load_data("../data/creditcard.csv")
    if df is not None:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        save_data(X_train, "../data/X_train.csv")
        save_data(X_test, "../data/X_test.csv")
        save_data(y_train, "../data/y_train.csv")
        save_data(y_test, "../data/y_test.csv")


if __name__ == "__main__":
    main()