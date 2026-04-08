import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib
# Import custom preprocessing pipeline wrapper
from src.data_preprocessing import create_preprocessing_pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

models = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
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
    elif model_code == "lgb":
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, min_data_in_leaf=20, min_gain_to_split=0.0, verbose=-1)
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
        pipeline = create_preprocessing_pipeline(model=raw_model, contamination=0.01, k_neighbors=5)
        
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

def objective(trial, model_code, X, y):
    params = {}
    
    if model_code == 'rf':
        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 400, step=100),
            'model__max_depth': trial.suggest_categorical('model__max_depth', [10, 20, 30, None]),
            'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 10),
            'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 4),
            'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None])
        }
    elif model_code == 'xgb':
        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 150, 400, step=50),
            'model__max_depth': trial.suggest_categorical('model__max_depth', [4, 6, 8]),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2, log=True),
            'model__subsample': trial.suggest_float('model__subsample', 0.6, 1.0),
            'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.6, 1.0),
            'model__gamma': trial.suggest_float('model__gamma', 0.0, 0.5)
        }
    elif model_code == 'lgb':
        params = {
            'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 300, step=100),
            'model__num_leaves': trial.suggest_categorical('model__num_leaves', [31, 50, 70]),
            'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2, log=True),
        }

    raw_model = initialize_models(model_code)
    pipe = create_preprocessing_pipeline(model=raw_model)
    pipe.set_params(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

def hyperparameter_tuning(X_train, y_train, models_to_tune):
    # Specify which models to tune to avoid tuning all 6 from scratch
    best_results_optuna = []
    best_models_optuna = {}
    print(f"Tuning model with Optuna...")

    for model_name, model_code in models_to_tune.items():
        print(f"Hyperparameter tuning for {model_name} using Optuna...")
        study = optuna.create_study(direction='maximize') # Maximize F1
        study.optimize(lambda trial: objective(trial, model_code, X_train, y_train), n_trials=10) # Reduced to 10 trials
        
        best_params = study.best_params
        raw_model = initialize_models(model_code)
        best_pipe = create_preprocessing_pipeline(model=raw_model, contamination=0.01, k_neighbors=5)
        best_pipe.set_params(**best_params)
        best_pipe.fit(X_train, y_train)
        
        best_models_optuna[model_code] = {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_estimator': best_pipe,
            'study': study
        }
        
        result_entry = {
            'Model Code': model_code,
            'Model Name': model_name,
            'Best F1': study.best_value,
        }
        best_results_optuna.append(result_entry)

    df_best_model_optuna = pd.DataFrame(best_results_optuna).sort_values(
        by="Best F1", ascending=False
    ).reset_index(drop=True)

    return df_best_model_optuna, best_models_optuna


def train_final_models(X_train, y_train, top_model_names):
    """
    Rebuild the pipeline and fit it directly on the entire training set for exporting.
    """
    fitted_models = {}
    for model_name in top_model_names:
        # Reverse lookup to get model_code
        model_code = next(code for code, name in models.items() if name == model_name)
        raw_model = initialize_models(model_code)
        pipeline = create_preprocessing_pipeline(model=raw_model, contamination=0.01, k_neighbors=5)
        
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