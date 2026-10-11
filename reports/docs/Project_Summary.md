# Project Summary

## Project Overview
A credit card fraud detection system using machine learning models to detect fraudulent transactions and prevent financial loss.
- Size: 284,807 transactions.
- Features: 31 features (28 PCA-transformed features, 'Amount', 'Time', 'Class').
- Class distribution: ~0.2% fraudulent transactions.
- Model trained: 7 algorithms (Random Forest, Gradient Boosting, XGBoost, LightGBM, K-Nearest Neighbors, Support Vector Machine, Gaussian Naive Bayes).
- Best performance on test set: Ensemble model with 80% F1-score (Random Forest ~81%).

## Deliverables Created
### 1. Data Preprocessing Module (`src/data_preprocessing.py`)
- Complete data loading.
- Removed duplicated transactions.
- 80/20 train/test split.
- Capped extreme values using Winsorizer method.
- RobustScaler for numerical features.
- Removed outliers of the normal class (Class = 0) using IsolationForest. Note: fraud rows are never dropped.
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

### 5. Business Analysis Module (`src/business_analysis.py`)
- Fraud amount caught / missed (`Amount` column) and genuine customers blocked for thresholds 0.05 – 0.9.
- Saves tables to `reports/business/` and trade-off charts to `reports/figures/`.

### 6. Streamlit Application (`app/app.py`)
- Interactive web application to test the trained models.
- Allows users to select a model and test it on random, fraud, or normal transactions.
- Displays transaction information and model predictions.

## Results

### Test Set Performance (Default Threshold = 0.5)

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
|-------|----------|--------|-----------|----------|---------|
| **Ensemble (Stacking)** | **0.9993** | **0.8000** | **0.8000** | **0.8000** | **0.9749** |
| Random Forest | 0.9994 | 0.7789 | 0.8409 | 0.8087 | 0.9652 |
| XGBoost | 0.9992 | 0.7895 | 0.7426 | 0.7653 | 0.9755 |
| LightGBM | 0.9993 | 0.7789 | 0.7957 | 0.7872 | 0.9680 |

### Selected Model: Stacking Ensemble
The Ensemble model is selected as the production model because:
- **Robust to the threshold:** It catches the same 76 of 95 frauds for any threshold from 0.05 to 0.8, with only 17–25 false alarms.
- **Balanced Precision / Recall (0.80 / 0.80):** Catches 80% of frauds and 80% of alerts are correct.
- **High ROC-AUC (0.9749):** Close to the best single model (XGBoost, 0.9755).
- Random Forest has a slightly higher F1 at 0.5 (0.8087), but its performance swings much more when the threshold changes.

### Key Observations
1. **Accuracy is misleading in this context:** All models achieve >99.9% accuracy simply because the dataset is 99.8% non-fraud. F1-score is the correct metric to evaluate performance on the minority (fraud) class.
2. **XGBoost has the lowest Precision (0.7426)** of the final models, meaning more false alarms (26 at threshold 0.5).
3. **Outlier removal must not touch the minority class:** Running IsolationForest on all rows removed 64% of training frauds; restricting it to Class = 0 raised Random Forest Recall from 0.53 to 0.78.
4. **The threshold is a business decision:** For Random Forest, lowering it from 0.5 to 0.05 catches 90% instead of 74% of the fraud amount but blocks 666 instead of 14 genuine customers.

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
  → OutlierRemover (IsolationForest on Class = 0 only, contamination=1%)
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
4. **Anomaly-based cleaning can delete the target:** In fraud detection the positive class *is* the anomaly, so outlier removal has to be restricted to the majority class (and its effect on the minority class should always be checked).
5. **F1-score is the right metric:** For highly imbalanced datasets like credit card fraud (~0.2% positive class), accuracy is misleading. F1-score properly captures the trade-off between catching fraud (Recall) and avoiding false alarms (Precision).

## Future Improvements
- **Feature engineering:** Create behavioral features (e.g., rolling transaction statistics, time-based patterns) to potentially push F1 above 0.80.
- **Advanced loss functions:** Experiment with Focal Loss for LightGBM/XGBoost to better handle class imbalance during training.
- **Anomaly detection features:** Use Autoencoder or Isolation Forest reconstruction error as an additional input feature.
- **Model monitoring:** Add drift detection to alert when incoming transaction patterns deviate from training data distribution.
- **API deployment:** Wrap the prediction pipeline in a REST API (FastAPI/Flask) for production integration.