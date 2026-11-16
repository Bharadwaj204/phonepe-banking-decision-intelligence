"""
part1.py
Part 1: Estimation, inference and regression baseline (25 marks).
"""

import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

OUTPUT_PATH = os.path.join("outputs", "part1_results.txt")


def format_value(val):
    if isinstance(val, bool):
        return "yes" if val else "no"
    elif isinstance(val, (int, np.integer)):
        return str(int(val))
    elif isinstance(val, (float, np.floating)):
        return f"{val:.4f}"
    return str(val)


def write_results(results_dict, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for k, v in results_dict.items():
            f.write(f"{k} = {format_value(v)}\n")


def run_part1():
    records = common.load_records()
    value = common.get_monthly_spend(records)
    X = common.get_X1_to_X7(records)
    X4 = X[:, 3]

    results = {}

    # 6.1 Estimation
    n = len(value)
    mean_value = float(np.mean(value))
    sd_value = common.sample_std(value)
    se_value = sd_value / math.sqrt(n)
    ci95_low = mean_value - 1.96 * se_value
    ci95_high = mean_value + 1.96 * se_value

    results["n"] = n
    results["mean_value"] = mean_value
    results["sd_value"] = sd_value
    results["se_value"] = se_value
    results["ci95_low"] = ci95_low
    results["ci95_high"] = ci95_high

    # 6.2 Hypothesis testing
    # H0: population mean = 8.0, H1: population mean != 8.0, alpha = 0.05
    z_stat = (mean_value - 8.0) / se_value
    reject_h0 = abs(z_stat) > 1.96
    results["z_stat"] = z_stat
    results["reject_h0"] = reject_h0

    # 6.3 Sample size
    # E = 0.5, n_required = ceil((1.96 * sd_value / 0.5)^2)
    n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)
    results["n_required"] = n_required

    # 6.4 Simple linear regression (monthly_spend on X4)
    x_bar = float(np.mean(X4))
    y_bar = float(np.mean(value))
    ss_xx = float(np.sum((X4 - x_bar) ** 2))
    ss_xy = float(np.sum((X4 - x_bar) * (value - y_bar)))
    reg_slope = ss_xy / ss_xx
    reg_intercept = y_bar - reg_slope * x_bar

    y_hat = reg_slope * X4 + reg_intercept
    tss = float(np.sum((value - y_bar) ** 2))
    ess = float(np.sum((y_hat - y_bar) ** 2))
    rss = float(np.sum((value - y_hat) ** 2))
    tss_equals_ess_plus_rss = abs(tss - (ess + rss)) < 1e-4
    r2 = 1.0 - (rss / tss)
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(value - y_hat)))

    results["reg_slope"] = reg_slope
    results["reg_intercept"] = reg_intercept
    results["tss"] = tss
    results["ess"] = ess
    results["rss"] = rss
    results["tss_equals_ess_plus_rss"] = tss_equals_ess_plus_rss
    results["r2"] = r2
    results["rmse"] = rmse
    results["mae"] = mae

    # 6.5 Gradient descent on standardised X4
    x4_sd = common.sample_std(X4)
    X4_std = (X4 - x_bar) / x4_sd

    m = 0.0
    c = 0.0
    lr = 0.1
    epochs = 500

    def compute_mse(m_val, c_val):
        preds = m_val * X4_std + c_val
        return float(np.mean((preds - value) ** 2))

    losses = {0: compute_mse(m, c)}
    for ep in range(1, epochs + 1):
        preds = m * X4_std + c
        residuals = preds - value
        grad_m = (2.0 / n) * float(np.dot(residuals, X4_std))
        grad_c = (2.0 / n) * float(np.sum(residuals))
        m -= lr * grad_m
        c -= lr * grad_c
        if ep in [1, 10, 100, 500]:
            losses[ep] = compute_mse(m, c)

    gd_slope = m / x4_sd
    gd_intercept = c - (m * x_bar / x4_sd)
    gd_slope_gap = abs(gd_slope - reg_slope)

    results["gd_J0"] = losses[0]
    results["gd_J1"] = losses[1]
    results["gd_J10"] = losses[10]
    results["gd_J100"] = losses[100]
    results["gd_J500"] = losses[500]
    results["gd_slope"] = gd_slope
    results["gd_intercept"] = gd_intercept
    results["gd_slope_gap"] = gd_slope_gap

    # 6.6 Funnel analysis and log-log regression
    sort_idx = common.stable_argsort(y_hat)
    raw_resids = value - y_hat
    resid_sd_lower_raw = common.sample_std(raw_resids[sort_idx[:400]])
    resid_sd_upper_raw = common.sample_std(raw_resids[sort_idx[400:]])
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    results["resid_sd_lower_raw"] = resid_sd_lower_raw
    results["resid_sd_upper_raw"] = resid_sd_upper_raw
    results["resid_sd_ratio_raw"] = resid_sd_ratio_raw

    # Regress ln(value) on ln(X4)
    ln_val = np.log(value)
    ln_x4 = np.log(X4)
    lx_bar = float(np.mean(ln_x4))
    ly_bar = float(np.mean(ln_val))
    ss_lxx = float(np.sum((ln_x4 - lx_bar) ** 2))
    ss_lxy = float(np.sum((ln_x4 - lx_bar) * (ln_val - ly_bar)))
    loglog_slope = ss_lxy / ss_lxx
    loglog_intercept = ly_bar - loglog_slope * lx_bar

    ll_y_hat = loglog_slope * ln_x4 + loglog_intercept
    ll_resids = ln_val - ll_y_hat
    ll_tss = float(np.sum((ln_val - ly_bar) ** 2))
    ll_rss = float(np.sum(ll_resids ** 2))
    loglog_r2 = 1.0 - (ll_rss / ll_tss)
    loglog_rmse = math.sqrt(ll_rss / n)

    ll_sort_idx = common.stable_argsort(ll_y_hat)
    resid_sd_lower_log = common.sample_std(ll_resids[ll_sort_idx[:400]])
    resid_sd_upper_log = common.sample_std(ll_resids[ll_sort_idx[400:]])
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    results["loglog_intercept"] = loglog_intercept
    results["loglog_slope"] = loglog_slope
    results["loglog_r2"] = loglog_r2
    results["loglog_rmse"] = loglog_rmse
    results["resid_sd_lower_log"] = resid_sd_lower_log
    results["resid_sd_upper_log"] = resid_sd_upper_log
    results["resid_sd_ratio_log"] = resid_sd_ratio_log

    # 6.7 Multiple regression: ln(value) on 1, ln(X4), X1, X2, X3, X5, X6, X7
    ones = np.ones((n, 1))
    col_ln_x4 = ln_x4.reshape(-1, 1)
    col_x1 = X[:, 0:1]
    col_x2 = X[:, 1:2]
    col_x3 = X[:, 2:3]
    col_x5 = X[:, 4:5]
    col_x6 = X[:, 5:6]
    col_x7 = X[:, 6:7]

    X_mat = np.hstack([ones, col_ln_x4, col_x1, col_x2, col_x3, col_x5, col_x6, col_x7])
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ ln_val)

    for i in range(8):
        results[f"mr_beta_{i}"] = float(beta[i])

    mr_y_hat = X_mat @ beta
    mr_tss = float(np.sum((ln_val - ly_bar) ** 2))
    mr_rss = float(np.sum((ln_val - mr_y_hat) ** 2))
    mr_r2 = 1.0 - (mr_rss / mr_tss)
    # p = 7 predictors for multiple regression
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
    # loglog has 1 predictor
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    results["mr_r2"] = mr_r2
    results["mr_adj_r2"] = mr_adj_r2
    results["loglog_adj_r2"] = loglog_adj_r2

    write_results(results)
    return results


def main():
    res = run_part1()
    print("Part 1 executed successfully.")
    print(f"n = {res['n']}")
    print(f"mean_value = {res['mean_value']:.4f}")
    print(f"gd_slope_gap = {res['gd_slope_gap']:.6f}")


if __name__ == "__main__":
    main()
