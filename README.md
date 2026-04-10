# Credit Card Fraud Detection 

![Figure](figures/r0zzc2r0zzc2r0zz.png)

This project is an end-to-end credit card fraud detection system that utilizes advanced machine learning techniques to identify fraudulent transactions and minimize financial loss. It features a robust data preprocessing pipeline, automated model tuning, ensemble learning, and an interactive web simulation.

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

## Key Features
- **End-to-End ML Pipeline**: From data ingestion and cleaning to model training, evaluation, and deployment.
- **Imbalanced Data Handling**: Uses Winsorizer for outlier capping, IsolationForest for anomaly removal, and SMOTE for minority class oversampling.
- **Comprehensive Evaluation**: Compares 7 different algorithms (Random Forest, XGBoost, LightGBM, etc.) with a focus on F1-Score.
- **Hyperparameter Tuning**: Automated tuning using Optuna for the top-performing models.
- **Stacking Ensemble**: Combines the strengths of the best models using a Logistic Regression meta-learner for maximum stability.
- **Interactive Web App**: A Streamlit application simulating a real-time transaction verification and alert system.

## Performance Highlights
The models were evaluated on a highly imbalanced dataset (only ~0.2% fraud instances). The **Stacking Ensemble** model achieved the best overall performance on the hold-out test set:
- **F1-Score**: 0.7416 
- **Precision**: 0.7952 (~80% of fraud alerts are correct)
- **Recall**: 0.6947 (~70% of actual frauds successfully blocked)
- **ROC-AUC**: 0.9618

For an in-depth analysis, please refer to the [Project Summary](reports/docs/Project_Summary.md).

## Project Structure

```bash
Credit-Card-Fraud-Detection/
├── data/                           # Data directory
│   └── creditcard.csv
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── model_training.py           # Model training and evaluation
│   ├── main_pipeline.py            # Main pipeline
│   ├── predict.py                  # Prediction module for new data
│   └── utils.py                    # Utility functions
├── app/                            # Streamlit web application
│   └── app.py            # Main Streamlit app
├── models/     
│   └── *.pkl                       # Saved 4 models including ensemble model and top 3 models
│                                   # according to F1 score
├── logs/                           # Log files
│   └── pipeline.log                # Pipeline log file
├── notebooks/                      # Jupyter notebooks for EDA
│   └── eda.ipynb
├── reports/
│   ├── figures/                # Figures for evaluation
│   │   └── *.png                       # Figures for evaluation including confusion matrix, ROC curve.
│   └── docs/                   # Project documentation
│       └── Project_Summary.md          # High-level project overview
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration file
└── README.md                       # This file
```

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

**Run the main pipeline:**

```bash
python src/main_pipeline.py
```

This will:
1. Load and preprocess the data.
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

**Run the Streamlit app:**

```bash
streamlit run app/app.py
```
