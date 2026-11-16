"""
part2.py
Part 2: Classification and evaluation (30 marks).
"""

import csv
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

RESULTS_TXT = os.path.join("outputs", "part2_results.txt")
THRESH_CSV = os.path.join("outputs", "part2_thresholds.csv")
KNN_CSV = os.path.join("outputs", "part2_knn_cv.csv")


def format_value(val):
    if isinstance(val, bool):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(int(val))
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    return str(val)


def write_results(results_dict, output_path=RESULTS_TXT):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for k, v in results_dict.items():
            f.write(f"{k} = {format_value(v)}\n")


def stratified_split(records):
    """
    Stratified split preserving original file order:
    For outcome 0: first floor(7 * count / 10) rows go to training.
    For outcome 1: first floor(7 * count / 10) rows go to training.
    Remaining go to testing.
    """
    y = common.get_loan_default(records)
    pos_idx = [i for i, val in enumerate(y) if val == 1]
    neg_idx = [i for i, val in enumerate(y) if val == 0]

    n_train_pos = math.floor(7 * len(pos_idx) / 10)
    n_train_neg = math.floor(7 * len(neg_idx) / 10)

    train_indices = sorted(pos_idx[:n_train_pos] + neg_idx[:n_train_neg])
    test_indices = sorted(pos_idx[n_train_pos:] + neg_idx[n_train_neg:])

    return train_indices, test_indices


def knn_predict(X_train_std, y_train, X_test_std, k):
    """
    KNN classification with Euclidean distance and stable tie-breaking.
    Predicts 1 if sum(labels) > k / 2 else 0.
    """
    predictions = np.empty(len(X_test_std), dtype=int)
    for i in range(len(X_test_std)):
        dists = np.sqrt(np.sum((X_train_std - X_test_std[i]) ** 2, axis=1))
        nearest_idx = np.argsort(dists, kind="stable")[:k]
        votes = y_train[nearest_idx]
        predictions[i] = 1 if np.sum(votes) > (k / 2.0) else 0
    return predictions


def run_part2():
    records = common.load_records()
    X = common.get_X1_to_X7(records)
    y = common.get_loan_default(records)

    train_idx, test_idx = stratified_split(records)

    X_train_raw = X[train_idx]
    y_train = y[train_idx]
    X_test_raw = X[test_idx]
    y_test = y[test_idx]

    # Standardise using only training statistics
    X_train_std, X_test_std, train_mu, train_sigma = common.standardise_train_test(
        X_train_raw, X_test_raw
    )

    results = {}
    results["train_rows"] = len(train_idx)
    results["train_positives"] = int(np.sum(y_train == 1))
    results["test_rows"] = len(test_idx)
    results["test_positives"] = int(np.sum(y_test == 1))

    # 7.2 Logistic regression from scratch
    # Initial weights and bias = 0. lr = 0.1, 1000 updates.
    n_tr = len(y_train)
    p = X_train_std.shape[1]
    w = np.zeros(p, dtype=float)
    b = 0.0
    lr = 0.1
    epochs = 1000

    def get_bce(weights, bias, X_data, y_data):
        prob = common.sigmoid(X_data @ weights + bias)
        return common.binary_cross_entropy(y_data, prob)

    losses = {0: get_bce(w, b, X_train_std, y_train)}

    for ep in range(1, epochs + 1):
        prob = common.sigmoid(X_train_std @ w + b)
        err = prob - y_train
        grad_w = (X_train_std.T @ err) / n_tr
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
        if ep in [1, 10, 100, 1000]:
            losses[ep] = get_bce(w, b, X_train_std, y_train)

    results["loss_epoch_0"] = losses[0]
    results["loss_epoch_1"] = losses[1]
    results["loss_epoch_10"] = losses[10]
    results["loss_epoch_100"] = losses[100]
    results["loss_epoch_1000"] = losses[1000]

    for i in range(1, 8):
        results[f"weight_X{i}"] = float(w[i - 1])
    results["bias"] = float(b)

    test_prob = common.sigmoid(X_test_std @ w + b)
    test_bce = common.binary_cross_entropy(y_test, test_prob)
    results["test_bce"] = test_bce

    # 7.3 Learning-rate comparison (200 updates from zero)
    for eta in [0.01, 0.1, 1.0]:
        w_eta = np.zeros(p, dtype=float)
        b_eta = 0.0
        for ep in range(1, 201):
            prob = common.sigmoid(X_train_std @ w_eta + b_eta)
            err = prob - y_train
            w_eta -= eta * (X_train_std.T @ err) / n_tr
            b_eta -= eta * float(np.mean(err))
        results[f"loss_epoch_200_eta_{eta}"] = get_bce(w_eta, b_eta, X_train_std, y_train)

    # 7.4 Odds ratios: exp(weight)
    for i in range(1, 8):
        results[f"odds_ratio_X{i}"] = float(np.exp(w[i - 1]))

    # 7.5 Threshold evaluation at 0.5 and 0.3
    for tau in [0.5, 0.3]:
        y_pred = (test_prob >= tau).astype(int)
        tp, fp, tn, fn = common.confusion_matrix(y_test, y_pred)
        acc = common.accuracy(tp, fp, tn, fn)
        prec = common.precision(tp, fp)
        rec = common.recall(tp, fn)
        f1 = common.f1_score(prec, rec)
        spec = common.specificity(tn, fp)
        prefix = f"tau_{tau}"
        results[f"{prefix}_tp"] = tp
        results[f"{prefix}_fp"] = fp
        results[f"{prefix}_tn"] = tn
        results[f"{prefix}_fn"] = fn
        results[f"{prefix}_accuracy"] = acc
        results[f"{prefix}_precision"] = prec
        results[f"{prefix}_recall"] = rec
        results[f"{prefix}_f1"] = f1
        results[f"{prefix}_specificity"] = spec

    # Blind accuracy (predicting all 0s)
    blind_tp, blind_fp, blind_tn, blind_fn = common.confusion_matrix(
        y_test, np.zeros(len(y_test), dtype=int)
    )
    results["blind_accuracy"] = common.accuracy(blind_tp, blind_fp, blind_tn, blind_fn)

    # 7.6 Threshold sweep (0.1 through 0.9 in increments of 0.1)
    os.makedirs(os.path.dirname(THRESH_CSV), exist_ok=True)
    with open(THRESH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        for tau_int in range(1, 10):
            tau = tau_int / 10.0
            y_pred = (test_prob >= tau).astype(int)
            tp, fp, tn, fn = common.confusion_matrix(y_test, y_pred)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            writer.writerow([f"{tau:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])

    # 7.7 KNN from scratch with 5-fold cross-validation
    k_candidates = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    fold_ids = np.arange(n_tr) % 5

    knn_cv_f1 = {}
    for k in k_candidates:
        fold_f1_scores = []
        for fold in range(5):
            val_mask = fold_ids == fold
            tr_mask = ~val_mask
            X_tr_f = X_train_std[tr_mask]
            y_tr_f = y_train[tr_mask]
            X_val_f = X_train_std[val_mask]
            y_val_f = y_train[val_mask]

            y_val_pred = knn_predict(X_tr_f, y_tr_f, X_val_f, k)
            tp, fp, tn, fn = common.confusion_matrix(y_val_f, y_val_pred)
            prec = common.precision(tp, fp)
            rec = common.recall(tp, fn)
            f1 = common.f1_score(prec, rec)
            fold_f1_scores.append(f1)
        knn_cv_f1[k] = float(np.mean(fold_f1_scores))

    with open(KNN_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        for k in k_candidates:
            writer.writerow([k, f"{knn_cv_f1[k]:.4f}"])

    # Select best K: highest mean F1, ties to smaller K
    best_k = min(k_candidates, key=lambda k: (-knn_cv_f1[k], k))

    # Refit KNN on all 560 training rows, evaluate on test
    knn_test_pred = knn_predict(X_train_std, y_train, X_test_std, best_k)
    k_tp, k_fp, k_tn, k_fn = common.confusion_matrix(y_test, knn_test_pred)
    k_acc = common.accuracy(k_tp, k_fp, k_tn, k_fn)
    k_prec = common.precision(k_tp, k_fp)
    k_rec = common.recall(k_tp, k_fn)
    k_f1 = common.f1_score(k_prec, k_rec)

    results["knn_best_k"] = best_k
    results["knn_tp"] = k_tp
    results["knn_fp"] = k_fp
    results["knn_tn"] = k_tn
    results["knn_fn"] = k_fn
    results["knn_accuracy"] = k_acc
    results["knn_precision"] = k_prec
    results["knn_recall"] = k_rec
    results["knn_f1"] = k_f1

    # 7.8 Cost-based threshold selection
    # Cost = 50 * FP + 400 * FN
    train_prob = common.sigmoid(X_train_std @ w + b)
    taus = [round(i * 0.05, 2) for i in range(1, 20)]  # 0.05 to 0.95

    best_tau = None
    min_train_cost = float("inf")

    for tau in taus:
        y_tr_pred = (train_prob >= tau).astype(int)
        tp, fp, tn, fn = common.confusion_matrix(y_train, y_tr_pred)
        cost = 50 * fp + 400 * fn
        if cost < min_train_cost:
            min_train_cost = cost
            best_tau = tau
        elif cost == min_train_cost and (best_tau is None or tau < best_tau):
            best_tau = tau

    # Test cost at best_tau
    test_pred_tau = (test_prob >= best_tau).astype(int)
    tp_t, fp_t, tn_t, fn_t = common.confusion_matrix(y_test, test_pred_tau)
    cost_test_at_tau = 50 * fp_t + 400 * fn_t

    # Test cost at 0.5
    test_pred_05 = (test_prob >= 0.5).astype(int)
    tp_05, fp_05, tn_05, fn_05 = common.confusion_matrix(y_test, test_pred_05)
    cost_test_at_05 = 50 * fp_05 + 400 * fn_05

    # Test cost flagging nobody (all 0s)
    # fp = 0, fn = count of positives in test
    cost_test_flag_nobody = 400 * int(np.sum(y_test == 1))

    results["cost_tau"] = float(best_tau)
    results["cost_train_at_tau"] = float(min_train_cost)
    results["cost_test_at_tau"] = float(cost_test_at_tau)
    results["cost_test_at_0.5"] = float(cost_test_at_05)
    results["cost_test_flag_nobody"] = float(cost_test_flag_nobody)

    write_results(results)
    return results, w, b, best_k, X_train_std, y_train, X_test_std, y_test, test_prob


def main():
    res, _, _, _, _, _, _, _, _ = run_part2()
    print("Part 2 executed successfully.")
    print(f"train_rows = {res['train_rows']}, test_rows = {res['test_rows']}")
    print(f"loss_epoch_0 = {res['loss_epoch_0']:.4f}")
    print(f"knn_best_k = {res['knn_best_k']}")
    print(f"cost_tau = {res['cost_tau']:.2f}")


if __name__ == "__main__":
    main()
