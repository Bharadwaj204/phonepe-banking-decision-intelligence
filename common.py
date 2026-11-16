"""
common.py
Shared data-loading and utility functions for PhonePe Banking Decision Intelligence capstone.
Follows all assignment requirements.
"""

import csv
import math
import os
import numpy as np

DATA_PATH = os.path.join("data", "records.csv")

FEATURE_COLS = [
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score",
    "irregular_salary_flag",
    "autopay_flag"
]
TARGET_COL = "loan_default"
VALUE_COL = "monthly_spend_inr_thousands"


def load_records(path=DATA_PATH):
    """Load records.csv and return list of dicts in original file order."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'. Run generate_data.py first.")
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_X1_to_X7(records):
    """Return numpy float array of shape (N, 7) for features X1 through X7."""
    # Support both domain column names and fallback X1..X7
    if FEATURE_COLS[0] in records[0]:
        cols = FEATURE_COLS
    else:
        cols = [f"X{i}" for i in range(1, 8)]
    return np.array([[float(r[col]) for col in cols] for r in records], dtype=float)


def get_X1_to_X5(records):
    """Return numpy float array of shape (N, 5) for features X1 through X5."""
    if FEATURE_COLS[0] in records[0]:
        cols = FEATURE_COLS[:5]
    else:
        cols = [f"X{i}" for i in range(1, 6)]
    return np.array([[float(r[col]) for col in cols] for r in records], dtype=float)


def get_loan_default(records):
    """Return integer array of shape (N,) for loan_default."""
    return np.array([int(r[TARGET_COL]) for r in records], dtype=int)


def get_monthly_spend(records):
    """Return float array of shape (N,) for monthly_spend_inr_thousands."""
    return np.array([float(r[VALUE_COL]) for r in records], dtype=float)


def sample_mean(arr):
    """Sample mean."""
    return float(np.mean(arr))


def sample_std(arr):
    """Sample standard deviation with Bessel's correction (ddof=1)."""
    return float(np.std(arr, ddof=1))


def standardise_train_test(X_train, X_test):
    """Standardise X_train and X_test using ONLY training mean and sample SD (ddof=1)."""
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0, ddof=1)
    # Avoid zero division if any constant feature
    sigma = np.where(sigma == 0, 1.0, sigma)
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    return X_train_std, X_test_std, mu, sigma


def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    """Binary cross entropy loss."""
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def confusion_matrix(y_true, y_pred):
    """Return (tp, fp, tn, fn) for binary classification."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def accuracy(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(prec, rec):
    return 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def stable_argsort(arr):
    """Stable argsort preserving original order for ties."""
    return np.argsort(arr, kind="stable")
