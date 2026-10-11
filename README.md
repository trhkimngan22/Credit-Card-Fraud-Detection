# Credit Card Fraud Detection 

![Banner](figures/banner.png)

This project is an end-to-end credit card fraud detection system that utilizes advanced machine learning techniques to identify fraudulent transactions and minimize financial loss. It features a robust data preprocessing pipeline, automated model tuning, ensemble learning, and an interactive web simulation.

## Project Structure

```bash
Credit-Card-Fraud-Detection/
├── data/                           # Data directory
│   ├── creditcard.csv              # Raw data (download from Kaggle, see above)
│   └── X_train.csv, X_test.csv, y_train.csv, y_test.csv   # Train/test split
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── model_training.py           # Model training and evaluation
│   ├── main_pipeline.py            # Main pipeline
│   ├── predict.py                  # Prediction module for new data
│   ├── business_analysis.py        # Fraud amount caught/missed per threshold
│   └── utils.py                    # Utility functions
├── app/                            # Streamlit web application
│   └── app.py            # Main Streamlit app
├── models/     
│   └── *.pkl                       # Saved 4 models including ensemble model and top 3 models
│                                   # according to F1 score
├── logs/                           # Created at runtime (log files are git-ignored)
│   └── pipeline.log                # Pipeline log file
├── notebooks/                      # Jupyter notebooks for EDA
│   └── eda.ipynb
├── reports/
│   ├── figures/                # Figures for evaluation
│   │   └── *.png                       # Confusion matrix, ROC curve, threshold trade-off
│   ├── business/               # Business analysis tables
│   │   └── *_thresholds.csv            # Fraud amount caught/missed per threshold
│   └── docs/                   # Project documentation
│       └── Project_Summary.md          # High-level project overview
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration file
└── README.md                       # This file
```

## Dataset Information
The dataset contains transactions made by credit cards in September 2013 by European cardholders (commonly known from the Kaggle dataset).
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (~0.172%)
- **Data Imbalance:** Highly imbalanced (Positive class accounts for less than 0.2% of all transactions).
- **Features:** 
  - `V1` to `V28`: Numerical features resulting from a PCA transformation (done to protect user confidentiality).
  - `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  - `Amount`: The original transaction amount.
  - `Class`: The target variable (0 for Normal, 1 for Fraud).

**Download:** The raw dataset is **not** included in a normal clone (`data/creditcard.csv` is stored with Git LFS and is ~150 MB). Download `creditcard.csv` from Kaggle — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — and place it at `data/creditcard.csv`.

## Key Features
- **End-to-End ML Pipeline**: From data ingestion and cleaning to model training, evaluation, and deployment.
- **Imbalanced Data Handling**: Uses Winsorizer for outlier capping, IsolationForest to remove noisy *normal* transactions (fraud rows are never dropped — see below), and SMOTE for minority class oversampling.
- **Comprehensive Evaluation**: Compares 7 different algorithms (Random Forest, XGBoost, LightGBM, etc.) with a focus on F1-Score.
- **Hyperparameter Tuning**: Automated tuning using Optuna for the top-performing models.
- **Stacking Ensemble**: Combines the strengths of the best models using a Logistic Regression meta-learner for maximum stability.
- **Business Analysis**: Translates model scores into money — fraud amount caught vs. missed and genuine customers blocked — across several decision thresholds.
- **Interactive Web App**: A Streamlit application simulating a real-time transaction verification and alert system.

## Performance Highlights
Hold-out test set: 56,746 transactions, 95 frauds worth 14,766 in total (`Amount`). Default threshold = 0.5.

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Stacking Ensemble | 0.8000 | 0.8000 | 0.8000 | 0.9749 |
| Random Forest | 0.8409 | 0.7789 | 0.8087 | 0.9652 |
| XGBoost | 0.7426 | 0.7895 | 0.7653 | 0.9755 |
| LightGBM | 0.7957 | 0.7789 | 0.7872 | 0.9680 |

The Stacking Ensemble catches 76 of 95 frauds (74% of the fraud amount) while wrongly blocking 19 genuine transactions out of ~56,650.

**Model selection (5-fold CV on the training set, all 7 models, after the outlier fix):**

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Random Forest | 0.8111 | 0.8202 | 0.8146 | 0.9693 |
| XGBoost | 0.7226 | 0.8307 | 0.7720 | 0.9779 |
| LightGBM | 0.5918 | 0.8413 | 0.6933 | 0.9820 |
| K-Nearest Neighbors | 0.4044 | 0.8386 | 0.5444 | 0.9240 |
| Support Vector Machine | 0.3617 | 0.7939 | 0.4967 | 0.9651 |
| Gaussian Naive Bayes | 0.3443 | 0.8333 | 0.4841 | 0.9747 |
| Gradient Boosting | 0.1279 | 0.8889 | 0.2232 | 0.9788 |

Random Forest, XGBoost and LightGBM are the top 3 models by F1. The other models reach similar Recall but far lower Precision, i.e. they raise many more false alarms. After Optuna tuning (10 trials each) the CV F1 is 0.8134 (RF), 0.8061 (LGB) and 0.7868 (XGB).

## Why outlier removal only touches normal transactions
Outlier removal is meant to clean noise from the data — e.g. data-entry errors or one-off unusual purchases by genuine customers — so the model learns a cleaner picture of "normal" behaviour. In fraud detection, however, the target class *is* the anomaly: fraudulent transactions are exactly the points an anomaly detector such as IsolationForest flags first. Running it on every row would therefore treat fraud as noise and delete the most informative positive examples (on this training set it would flag **242 of the 378 frauds, 64%**), leaving the model to learn only from the hardest, least typical fraud cases.

For this reason `OutlierRemover` fits IsolationForest on `Class = 0` rows only and removes outliers only among them; every fraud row is kept (`outlier_majority_only: true` in `config.yaml`). This gives two benefits:
- **Cleaner majority class:** unusual genuine transactions that look like fraud are removed, which reduces false alarms.
- **No information loss on the minority class:** with only ~0.17% fraud, every positive example matters, and SMOTE then oversamples from the full set of real frauds.

## Business View: Choosing a Threshold
A model outputs a fraud probability; the business decides the cut-off. Lowering the threshold blocks more fraud but also more genuine customers (false alarms → support calls, card declines, churn). `src/business_analysis.py` reports, for thresholds from 0.05 to 0.9, how much fraud money is caught/missed and how many genuine customers are blocked. Full tables: `reports/business/*_thresholds.csv`; charts: `reports/figures/*_threshold_tradeoff.png`.

Random Forest on the test set:

| Threshold | Fraud caught | Fraud missed | Fraud amount caught | Fraud amount missed | Genuine customers blocked |
|-----------|-------------|-------------|--------------------|--------------------|--------------------------|
| 0.05 | 84 | 11 | 13,266 (90%) | 1,501 | 666 |
| 0.10 | 80 | 15 | 11,708 (79%) | 3,058 | 212 |
| 0.30 | 77 | 18 | 10,984 (74%) | 3,783 | 25 |
| 0.50 | 74 | 21 | 10,958 (74%) | 3,808 | 14 |
| 0.70 | 69 | 26 | 8,893 (60%) | 5,874 | 9 |
| 0.90 | 59 | 36 | 5,771 (39%) | 8,995 | 8 |

Takeaways:
- **Between 0.3 and 0.5 the trade-off is cheap:** the fraud amount caught barely changes while false alarms drop from 25 to 14.
- **Above 0.6 savings drop fast:** the frauds that slip through are the larger ones (fraud amount caught falls from 74% to 39% at 0.9).
- **Going from 0.5 down to 0.05** recovers 2,307 more fraud money but blocks 652 more genuine customers, so it only pays off if one false alarm costs less than ~3.5 per customer. The cost of a false alarm is a business assumption (`false_positive_cost` in `config.yaml`, default 5); the `Net savings` column = fraud amount caught − cost × genuine customers blocked.
- The Stacking Ensemble is much less sensitive to the threshold: it catches the same 76 frauds anywhere from 0.05 to 0.8, while false alarms only go from 25 to 17, so a threshold of 0.5–0.8 is a safe operating point.

For an in-depth analysis, please refer to the [Project Summary](reports/docs/Project_Summary.md).

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**Prepare the data** (after downloading `data/creditcard.csv` from Kaggle):

```bash
python -m src.data_preprocessing
```

This removes duplicates and writes the stratified 80/20 train/test split to `data/`.

**Run the main pipeline:**

```bash
python src/main_pipeline.py
```

This will:
1. Load the training split.
2. Train 7 different models.
3. Evaluate the models and select the best top 3 models.
4. Apply hyperparameter tuning for the top 3 models.
5. Train an ensemble model using the best model.
6. Save the models to `models/` directory.

**Run the prediction module:**

To make predictions on new data:

```bash
python src/predict.py
```

This will:
1. Load the latest trained model.
2. Preprocess the input data.
3. Make prediction.
4. Save evaluation plots to `reports/figures/` directory.
5. Write log to `logs/pipeline.log` file.

**Run the business analysis:**

```bash
python src/business_analysis.py
```

This writes a threshold table per model to `reports/business/` and a trade-off chart to `reports/figures/`.

**Run the Streamlit app:**

```bash
streamlit run app/app.py
```
