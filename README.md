# Credit Card Fraud Detection 

This project is a credit card fraud detection system that uses machine learning to detect fraudulent transactions.

## Project Structure

Credit-Card-Fraud-Detection/
├── data/                           # Data directory
│   └── creditcard.csv
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── model_training.py           # Model training and evaluation
│   ├── main_pipeline.py            # Main pipeline
│   └── predict.py                  # Prediction module for new data
├── app/                            # Streamlit web application
│   ├── app.py            # Main Streamlit app
│   ├── README.md                   # App documentation
│   └── TROUBLESHOOTING.md          # App troubleshooting guide
├── models/     
│   └── *.pkl                       # Saved 4 models including ensemble model and top 3 models
│                                   # according to F1 score
├── notebooks/                      # Jupyter notebooks for EDA
│   └── eda.ipynb
├── tests/                          # Unit tests
│   └── test.py
├── docs/                           # Project documentation
│   ├── PROJECT_SUMMARY.md          # High-level project overview
│   ├── FINAL_OVERVIEW.md           # Quick reference guide
│   ├── FIXES_APPLIED.md            # Bug fixes and solutions
│   ├── CI_CD_GUIDE.md              # CI/CD pipeline documentation
│   ├── CI_CD_SUMMARY.md            # CI/CD implementation summary
│   ├── CI_CD_FIXES.md              # CI/CD fixes applied
│   └── CI_CD_FINAL_SUMMARY.md      # Final CI/CD summary
├── requirements.txt                # Python dependencies
└── README.md                       # This file