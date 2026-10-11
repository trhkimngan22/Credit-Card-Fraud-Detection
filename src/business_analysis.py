import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logger, load_config, ROOT_DIR

config = load_config()
logger = setup_logger(__name__)

THRESHOLDS = config['business']['thresholds']
FP_COST = config['business']['false_positive_cost']


def threshold_report(y_true, y_prob, amount, thresholds=THRESHOLDS, fp_cost=FP_COST):
    """
    Translate model scores into business numbers for each decision threshold.

    A transaction is blocked when its fraud probability >= threshold.
        - Fraud caught / missed: count and Amount of fraud blocked (TP) / let through (FN).
        - Legit blocked: genuine customers wrongly blocked (FP) and the Amount they tried to spend.
        - Net savings vs. having no model: fraud Amount caught - fp_cost * FP,
          where fp_cost is an ASSUMED cost per false alarm (review call, customer friction).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amount = np.asarray(amount)
    rows = []
    for t in thresholds:
        blocked = y_prob >= t
        tp = blocked & (y_true == 1)
        fp = blocked & (y_true == 0)
        fn = ~blocked & (y_true == 1)
        n_fraud = int((y_true == 1).sum())
        caught_amt = amount[tp].sum()
        missed_amt = amount[fn].sum()
        rows.append({
            "Threshold": t,
            "Blocked": int(blocked.sum()),
            "Fraud caught": int(tp.sum()),
            "Fraud missed": int(fn.sum()),
            "Legit blocked": int(fp.sum()),
            "Precision": tp.sum() / blocked.sum() if blocked.sum() else np.nan,
            "Recall": tp.sum() / n_fraud if n_fraud else np.nan,
            "Fraud amount caught": round(caught_amt, 2),
            "Fraud amount missed": round(missed_amt, 2),
            "% fraud amount caught": caught_amt / (caught_amt + missed_amt) if (caught_amt + missed_amt) else np.nan,
            "Legit amount blocked": round(amount[fp].sum(), 2),
            "Net savings": round(caught_amt - fp_cost * fp.sum(), 2),
        })
    return pd.DataFrame(rows)


def plot_threshold_tradeoff(report, model_name, reports_dir):
    """Plot fraud amount missed vs. legit customers blocked across thresholds."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(report["Threshold"], report["Fraud amount missed"], marker="o", color="tab:red", label="Fraud amount missed")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Fraud amount missed", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(report["Threshold"], report["Legit blocked"], marker="s", color="tab:blue", label="Legit customers blocked")
    ax2.set_ylabel("Legit customers blocked", color="tab:blue")
    plt.title(f"Threshold trade-off of {model_name}")
    fig.tight_layout()

    os.makedirs(reports_dir, exist_ok=True)
    save_path = os.path.join(reports_dir, f"{model_name}_threshold_tradeoff.png")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved threshold trade-off plot to {save_path}")


def main():
    logger.info("Starting business analysis...")

    X_test = pd.read_csv(os.path.join(ROOT_DIR, config['paths']['test_data']))
    y_test = pd.read_csv(os.path.join(ROOT_DIR, config['paths']['test_label'])).squeeze()
    models_dir = os.path.join(ROOT_DIR, config['paths']['models_dir'])
    reports_dir = os.path.join(ROOT_DIR, config['paths']['reports_dir'])
    tables_dir = os.path.join(ROOT_DIR, config['paths']['business_dir'])
    os.makedirs(tables_dir, exist_ok=True)

    total_fraud_amt = X_test.loc[y_test == 1, "Amount"].sum()
    logger.info(f"Test set: {int(y_test.sum())} frauds worth {total_fraud_amt:,.2f} in total")

    for model_file in sorted(f for f in os.listdir(models_dir) if f.endswith('.pkl')):
        model_name = model_file.split('.')[0]
        pipeline = joblib.load(os.path.join(models_dir, model_file))
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        report = threshold_report(y_test, y_prob, X_test["Amount"])
        report.to_csv(os.path.join(tables_dir, f"{model_name}_thresholds.csv"), index=False)
        logger.info(f"Threshold report for {model_name}:\n{report.to_string(index=False)}")
        plot_threshold_tradeoff(report, model_name, reports_dir)


if __name__ == "__main__":
    main()
