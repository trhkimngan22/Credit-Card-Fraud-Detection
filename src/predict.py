import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logger, load_config, ROOT_DIR

config = load_config()
logger = setup_logger(__name__)

def load_pipeline (pipeline_path):
    pipeline = joblib.load(pipeline_path)
    return pipeline

def predict_fraud (pipeline, X_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def evaluate_model (y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return accuracy, recall, precision, f1, roc_auc

def print_report (model_name, accuracy, recall, precision, f1, roc_auc):
    logger.info(f"Model Performance Report")
    logger.info(f"Evaluation for {model_name} completed")
    logger.info(f"Results for {model_name}:")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")

def plot_and_save_confusion_matrix(y_true, y_pred, model_name, reports_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix of {model_name}')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    
    thresh = cm.max() / 2.
    import itertools
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    
    safe_name = model_name.replace(" ", "_").lower()
    save_path = os.path.join(reports_dir, f"{safe_name}_cm.png")
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")

def plot_and_save_roc_curve(pipeline, X_test, y_test, model_name, reports_dir):
    safe_name = model_name.replace(" ", "_").lower()
    save_path = os.path.join(reports_dir, f"{safe_name}_roc.png")
    os.makedirs(reports_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=model_name)
    plt.title(f'ROC Curve of {model_name}')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")

def main():
    logger.info("Starting prediction pipeline...")
    
    # Paths for data and models
    test_data_path = os.path.join(ROOT_DIR, config['paths']['test_data'])
    test_label_path = os.path.join(ROOT_DIR, config['paths']['test_label'])
    models_dir = os.path.join(ROOT_DIR, config['paths']['models_dir'])
    reports_dir = os.path.join(ROOT_DIR, config['paths']['reports_dir'])
    
    # Load data
    try:
        X_test = pd.read_csv(test_data_path)
        y_test = pd.read_csv(test_label_path).squeeze()
    except FileNotFoundError:
        logger.error(f"Training data not found at {test_data_path}. Please run data_preprocessing.py first.")
        return
    
    # Load models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.split('.')[0]
        pipeline = load_pipeline(model_path)
        y_pred, y_prob = predict_fraud(pipeline, X_test)
        accuracy, recall, precision, f1, roc_auc = evaluate_model(y_test, y_pred, y_prob)
        print_report(model_name, accuracy, recall, precision, f1, roc_auc)
        
        # Save plots
        plot_and_save_confusion_matrix(y_test, y_pred, model_name, reports_dir)
        plot_and_save_roc_curve(pipeline, X_test, y_test, model_name, reports_dir)

if __name__ == "__main__":
    main()