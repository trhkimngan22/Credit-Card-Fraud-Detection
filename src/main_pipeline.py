import os
import sys
# Add project root to path so 'from src.xxx import yyy' works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_training import train_all_models, select_top3_model, train_final_models, save_model, hyperparameter_tuning
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def format_results(results):
    formatted_list = []
    import numpy as np
    for model_name, cv_results in results.items():
        formatted_list.append({
            "Model": model_name,
            "Accuracy": np.mean(cv_results["test_accuracy"]),
            "Precision": np.mean(cv_results["test_precision"]),
            "Recall": np.mean(cv_results["test_recall"]),
            "F1-Score": np.mean(cv_results["test_f1"]),
            "ROC-AUC": np.mean(cv_results["test_roc_auc"])
        })

    return pd.DataFrame(formatted_list)
        

def main():
    """Main execution block to run the complete training flow."""
    logger.info("Starting model training pipeline...")
    
    # Paths for data and models
    train_data_path = "../data/X_train.csv"
    train_label_path = "../data/y_train.csv"
    models_dir = "../models/"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    try:
        logger.info("Loading training data...")
        X_train = pd.read_csv(train_data_path)
        y_train = pd.read_csv(train_label_path).squeeze()
    except FileNotFoundError:
        logger.error(f"Training data not found at {train_data_path}. Please run data_preprocessing.py first.")
        return

    # Phase 1: Train all models via CV
    cv_results = train_all_models(X_train, y_train)

    # Phase 2: Select top 3 models
    top3_names = select_top3_model(cv_results)
    df_results = format_results(cv_results)
    df_results = df_results.sort_values(by="F1-Score", ascending=False, ignore_index=True)
    print(df_results)

    # Phase 3: Hyperparameters tuning
    models_to_tune = {"Random Forest": "rf", "K-Nearest Neighbors": "knn", "XGBoost": "xgb"}
    df_best_models, best_models_optuna = hyperparameter_tuning(X_train, y_train, models_to_tune)
    print(df_best_models)
    print(best_models_optuna)
        

    # # Phase 4: Save finalized models
    # save_model(final_models, models_dir)
    
    # logger.info("Model training pipeline completed successfully!")

if __name__ == "__main__":
    main()