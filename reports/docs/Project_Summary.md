# Project Summary

## Project Overview
A credit card fraud detection system using machine learning models to detect fraudulent transactions and prevent financial loss.
- Size: 284,807 transactions.
- Features: 30 features (28 PCA-transformed features, 'Amount', 'Class').
- Class distribution: ~0.2% fraudulent transactions.
- Model trained: 7 algorithms (Random Forest, Gradient Boosting, XGBoost, LightGBM, K-Nearest Neighbors, Support Vector Machine, Gaussian Naive Bayes).
- Best performance on test set: Ensemble model with ~74% F1-score.

## Deliverables Created
### 1. Data Preprocessing Module (`src/data_preprocessing.py`)
- Complete data loading.
- Removed duplicated transactions.
- 80/20 train/test split.
- Capped extreme values using Winsorizer method.
- StandardScaler for numerical features.
- Removed outliers using IsolationForest.
- Applied SMOTE to oversample the minority class.

### 2. Model Training Module (`src/model_training.py`)
- 7 ML models implemented:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - K-Nearest Neighbors
    - Support Vector Machine
    - Gaussian Naive Bayes
- Evaluation metrics for each model: Accuracy, Precision, Recall, F1 Score, ROC AUC.
- Model selection based on F1 score (best for imbalanced data).
- Hyperparameter tuning for top3 highest F1-score models using Optuna.
- Ensemble model using Stacking method.

### 3. Main Pipeline (`src/main_pipeline.py`)
Orchestrates the complete workflow:
```bash
Data preprocessing -> Model training -> Model evaluation -> Model selection -> Hyperparameter tuning -> Ensemble model -> Save models to `models/` directory.
```

### 4. Predict Module (`src/predict.py`)
- Predict fraud or normal transactions.
- Save performance prediction figures to `reports/figures/` directory.

### 5. Streamlit Application (`app/app.py`)
- Interactive web application to test the trained models.
- Allows users to select a model and test it on random, fraud, or normal transactions.
- Displays transaction information and model predictions.

## Results

### Test Set Performance (Default Threshold = 0.5)

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
|-------|----------|--------|-----------|----------|---------|
| **Ensemble (Stacking)** | **0.9992** | **0.6947** | **0.7952** | **0.7416** | **0.9618** |
| XGBoost | 0.9991 | 0.7368 | 0.7216 | 0.7292 | 0.9617 |
| LightGBM | 0.9991 | 0.6842 | 0.7647 | 0.7222 | 0.9631 |
| Random Forest | 0.9989 | 0.5263 | 0.7353 | 0.6135 | 0.9578 |

### Selected Model: Stacking Ensemble
The Ensemble model was selected as the final production model because:
- **Highest F1-score (0.7416):** Best balance between Precision and Recall across all models.
- **Highest Precision (0.7952):** ~80% of fraud alerts are correct, minimizing false alarms that inconvenience legitimate customers.
- **Strong Recall (0.6947):** Captures ~70% of actual fraud cases.
- **Top-tier ROC-AUC (0.9618):** Demonstrates excellent class separation ability.

### Key Observations
1. **Accuracy is misleading in this context:** All models achieve >99.9% accuracy simply because the dataset is 99.8% non-fraud. F1-score is the correct metric to evaluate performance on the minority (fraud) class.
2. **XGBoost has the highest Recall (0.7368)** but at the cost of lower Precision (0.7216), meaning more false alarms.
3. **Random Forest underperforms significantly** with only 52.6% Recall — nearly half of all fraud cases are missed.
4. **The Stacking Ensemble combines the strengths** of XGBoost (high Recall) and LightGBM (high Precision) through a LogisticRegression meta-learner, resulting in the most balanced and robust predictions.

## Technical Architecture

### Data Pipeline
```
Raw CSV (284,807 rows)
  → Remove duplicates
  → 80/20 stratified train/test split
  → Save to CSV (X_train, X_test, y_train, y_test)
```

### Training Pipeline (inside imblearn Pipeline, per CV fold)
```
X_train fold
  → Winsorizer (cap extreme values, IQR method)
  → RobustScaler (standardize features)
  → OutlierRemover (IsolationForest, contamination=1%)
  → SMOTE (oversample minority class)
  → Classifier (RF / XGB / LGB / etc.)
```

### Prediction Pipeline
```
Saved .pkl model
  → Load with joblib
  → predict(X_test) → y_pred
  → predict_proba(X_test) → y_prob
  → Evaluate metrics
  → Save confusion matrix & ROC curve plots
```

## Lessons Learned
1. **Data leakage prevention is critical:** Wrapping all preprocessing (scaling, outlier removal, SMOTE) inside the `imblearn.Pipeline` ensures that test data is never used during training transformations.
2. **SMOTE is only applied during training:** The `imblearn.Pipeline` automatically bypasses samplers (OutlierRemover, SMOTE) during `predict()` calls, ensuring test data integrity.
3. **Ensemble methods provide stability:** While individual models may excel in one metric (e.g., XGBoost in Recall), the Stacking Ensemble delivers the most consistent and balanced performance.
4. **F1-score is the right metric:** For highly imbalanced datasets like credit card fraud (~0.2% positive class), accuracy is misleading. F1-score properly captures the trade-off between catching fraud (Recall) and avoiding false alarms (Precision).

## Future Improvements
- **Feature engineering:** Create behavioral features (e.g., rolling transaction statistics, time-based patterns) to potentially push F1 above 0.80.
- **Advanced loss functions:** Experiment with Focal Loss for LightGBM/XGBoost to better handle class imbalance during training.
- **Anomaly detection features:** Use Autoencoder or Isolation Forest reconstruction error as an additional input feature.
- **Model monitoring:** Add drift detection to alert when incoming transaction patterns deviate from training data distribution.
- **API deployment:** Wrap the prediction pipeline in a REST API (FastAPI/Flask) for production integration.