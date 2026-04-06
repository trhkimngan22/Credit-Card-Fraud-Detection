import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
import joblib
# Import custom preprocessing pipeline wrapper
from src.data_preprocessing import create_preprocessing_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

models = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "gnb": "Gaussian Naive Bayes"
}

def initialize_models(model_code: str):
    """
    Initialize base models to be trained.
    """
    if model_code == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_code == "gb":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_code == "xgb":
        return XGBClassifier(random_state=RANDOM_STATE)
    elif model_code == "knn":
        return KNeighborsClassifier()
    elif model_code == "svm":
        return SVC(random_state=RANDOM_STATE)
    elif model_code == "gnb":
        return GaussianNB()
    else:
        logger.error(f"Invalid model code: {model_code}")
        return None

def train_model(pipeline, X_train, y_train):
    """
    Evaluate modeling pipeline using cross-validation.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    return results

def train_all_models(X_train, y_train):
    """
    Wrap all models in the preprocessing pipeline, evaluate them via cross validation, and return results.
    """
    results = {}
    for model_code, model_name in models.items():
        logger.info(f"Evaluating {model_name} with CV...")
        raw_model = initialize_models(model_code)
        if raw_model is None:
            continue
            
        # Wrap the model in our data preprocessing steps
        pipeline = create_preprocessing_pipeline(model=raw_model)
        
        try:
            cv_results = train_model(pipeline, X_train, y_train)
            results[model_name] = cv_results
            logger.info(f"Evaluation for {model_name} completed")
            logger.info(f"Results for {model_name}:")
            logger.info(f"  Accuracy: {np.mean(cv_results['test_accuracy']):.4f} (+/- {np.std(cv_results['test_accuracy']):.4f})")
            logger.info(f"  Recall: {np.mean(cv_results['test_recall']):.4f} (+/- {np.std(cv_results['test_recall']):.4f})")
            logger.info(f"  Precision: {np.mean(cv_results['test_precision']):.4f} (+/- {np.std(cv_results['test_precision']):.4f})")
            logger.info(f"  F1-score: {np.mean(cv_results['test_f1']):.4f} (+/- {np.std(cv_results['test_f1']):.4f})")
            logger.info(f"  ROC-AUC: {np.mean(cv_results['test_roc_auc']):.4f} (+/- {np.std(cv_results['test_roc_auc']):.4f})")
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    return results

def select_top3_model(results):
    """
    Evaluate CV results and return the names of the top 3 models based on F1-score.
    """
    model_scores = []
    for model_name, cv_results in results.items():
        mean_score = np.mean(cv_results['test_f1'])
        model_scores.append((model_name, mean_score))
        
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top3_models = model_scores[:3]
    
    logger.info("Top 3 models:")
    top3_names = []
    for rank, (model_name, score) in enumerate(top3_models, start=1):
        logger.info(f"  {rank}. {model_name} with F1-score: {score:.4f}")
        top3_names.append(model_name)
        
    return top3_names

def train_final_models(X_train, y_train, top_model_names):
    """
    Rebuild the pipeline and fit it directly on the entire training set for exporting.
    """
    fitted_models = {}
    for model_name in top_model_names:
        # Reverse lookup to get model_code
        model_code = next(code for code, name in models.items() if name == model_name)
        
        raw_model = initialize_models(model_code)
        pipeline = create_preprocessing_pipeline(model=raw_model)
        
        logger.info(f"Fitting final pipeline for {model_name} on entire train set...")
        pipeline.fit(X_train, y_train)
        
        fitted_models[model_name] = pipeline
        
    return fitted_models

def save_model(models_dict, file_path):
    """
    Save trained model pipelines to file.
    models_dict is typically the output of train_final_models().
    """
    # Ensure there's a trailing slash or separator if file_path is meant to be a prefix
    if not file_path.endswith('/') and not file_path.endswith('_'):
        file_path += "_"
        
    for model_name, pipeline in models_dict.items():
        try:
            safe_name = model_name.replace(" ", "_").lower()
            full_path = file_path + safe_name + ".pkl"
            joblib.dump(pipeline, full_path)
            logger.info(f"Model pipeline saved to {full_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")