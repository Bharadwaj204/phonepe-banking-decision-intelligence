"""
Part 4: split quality, voting, AdaBoost, out-of-bag probability, and parameter counts.
"""

import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import common
import part2

RESULTS_PATH = os.path.join(ROOT, "outputs", "part4_results.txt")
SPLITS_PATH = os.path.join(ROOT, "outputs", "part4_splits.csv")
VOTING_PATH = os.path.join(ROOT, "outputs", "part4_voting.csv")
ADABOOST_PATH = os.path.join(ROOT, "outputs", "part4_adaboost.csv")
ARCHITECTURE_PATH = os.path.join(ROOT, "outputs", "part4_architecture.csv")

K_CANDIDATES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]


def binary_entropy(labels):
    if len(labels) == 0:
        return 0.0
    positive_rate = float(np.mean(labels))
    if positive_rate == 0.0 or positive_rate == 1.0:
        return 0.0
    negative_rate = 1.0 - positive_rate
    return -(
        positive_rate * math.log2(positive_rate)
        + negative_rate * math.log2(negative_rate)
    )


def binary_gini(labels):
    if len(labels) == 0:
        return 0.0
    positive_rate = float(np.mean(labels))
    negative_rate = 1.0 - positive_rate
    return 1.0 - positive_rate**2 - negative_rate**2


def evaluate_split(name, mask, labels):
    yes_labels = labels[mask]
    no_labels = labels[~mask]
    total = len(labels)
    root_entropy = binary_entropy(labels)
    root_gini = binary_gini(labels)
    weighted_entropy = (
        len(yes_labels) * binary_entropy(yes_labels)
        + len(no_labels) * binary_entropy(no_labels)
    ) / total
    weighted_gini = (
        len(yes_labels) * binary_gini(yes_labels)
        + len(no_labels) * binary_gini(no_labels)
    ) / total
    return {
        "split": name,
        "mask": mask,
        "n_yes": len(yes_labels),
        "positives_yes": int(np.sum(yes_labels)),
        "n_no": len(no_labels),
        "positives_no": int(np.sum(no_labels)),
        "info_gain": root_entropy - weighted_entropy,
        "gini_gain": root_gini - weighted_gini,
    }


def fit_logistic_regression(X_train, y_train):
    """Match the Part 2 Task 2.2 zero-init batch-gradient-descent model."""
    n_train, n_features = X_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    learning_rate = 0.1
    for _ in range(1000):
        probabilities = common.sigmoid(X_train @ weights + bias)
        errors = probabilities - y_train
        weights -= learning_rate * (X_train.T @ errors) / n_train
        bias -= learning_rate * float(np.mean(errors))
    return weights, bias


def select_knn_k(X_train, y_train):
    """Match the Part 2 Task 2.7 five-fold CV and tie-break convention."""
    fold_ids = np.arange(len(y_train)) % 5
    mean_f1_by_k = {}
    for k in K_CANDIDATES:
        fold_f1_scores = []
        for fold in range(5):
            validation_mask = fold_ids == fold
            training_mask = ~validation_mask
            validation_predictions = part2.knn_predict(
                X_train[training_mask],
                y_train[training_mask],
                X_train[validation_mask],
                k,
            )
            tp, fp, tn, fn = common.confusion_matrix(
                y_train[validation_mask], validation_predictions
            )
            precision = common.precision(tp, fp)
            recall = common.recall(tp, fn)
            fold_f1_scores.append(common.f1_score(precision, recall))
        mean_f1_by_k[k] = float(np.mean(fold_f1_scores))
    return min(K_CANDIDATES, key=lambda k: (-mean_f1_by_k[k], k))


def knn_test_probabilities(X_train, y_train, X_test, k):
    probabilities = np.empty(len(X_test), dtype=float)
    predictions = np.empty(len(X_test), dtype=int)
    required_votes = (k + 1) // 2
    for row_index, row in enumerate(X_test):
        distances = np.sqrt(np.sum((X_train - row) ** 2, axis=1))
        nearest_indices = np.argsort(distances, kind="stable")[:k]
        positive_votes = int(np.sum(y_train[nearest_indices]))
        probabilities[row_index] = positive_votes / k
        predictions[row_index] = int(positive_votes >= required_votes)
    return probabilities, predictions


def confusion_metrics(y_true, y_pred):
    tp, fp, tn, fn = common.confusion_matrix(y_true, y_pred)
    precision = common.precision(tp, fp)
    recall = common.recall(tp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": common.f1_score(precision, recall),
    }


def verify_part2_consistency(logistic_metrics, knn_metrics, best_k):
    """Catch accidental drift from the already-saved Part 2 results."""
    if not os.path.exists(os.path.join(ROOT, "outputs", "part2_results.txt")):
        return
    saved = {}
    with open(
        os.path.join(ROOT, "outputs", "part2_results.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            key, value = line.strip().split(" = ", 1)
            saved[key] = value

    checks = [
        ("knn_best_k", best_k),
        *(
            (f"tau_0.3_{name}", logistic_metrics[name])
            for name in ("tp", "fp", "tn", "fn")
        ),
        *(
            (f"knn_{name}", knn_metrics[name])
            for name in ("tp", "fp", "tn", "fn")
        ),
    ]
    for key, actual in checks:
        if key in saved and int(saved[key]) != int(actual):
            raise ValueError(
                f"Part 4 {key}={actual} does not match saved Part 2 value {saved[key]}."
            )


def run_adaboost(results):
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=int)
    hypotheses = (
        np.array([1, 1, -1, -1, -1, -1, -1, -1], dtype=int),
        np.array([-1, 1, 1, -1, 1, -1, 1, -1], dtype=int),
    )
    weights = np.full(len(y), 1.0 / len(y), dtype=float)
    csv_rows = []
    alphas = []

    for round_index, hypothesis in enumerate(hypotheses, start=1):
        weights_before = weights.copy()
        error = float(np.sum(weights_before[hypothesis != y]))
        if error <= 0.0 or error >= 1.0:
            raise ValueError(f"AdaBoost round {round_index} has invalid error {error}.")
        alpha = 0.5 * math.log((1.0 - error) / error)
        unnormalised = weights_before * np.exp(-alpha * y * hypothesis)
        weights = unnormalised / np.sum(unnormalised)
        if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ArithmeticError(
                f"AdaBoost round {round_index} weights did not normalise to one."
            )
        alphas.append(alpha)

        if round_index == 1:
            results["round1_error"] = error
            results["round1_alpha"] = alpha
        else:
            results["round2_error"] = error
            results["round2_alpha"] = alpha

        for point_index in range(len(y)):
            csv_rows.append(
                [
                    round_index,
                    point_index + 1,
                    f"{weights_before[point_index]:.4f}",
                    f"{unnormalised[point_index]:.4f}",
                    f"{weights[point_index]:.4f}",
                ]
            )

    ensemble_score = alphas[0] * hypotheses[0] + alphas[1] * hypotheses[1]
    ensemble_prediction = np.sign(ensemble_score)
    results["ensemble_accuracy"] = float(np.mean(ensemble_prediction == y))

    with open(ADABOOST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["round", "point", "weight_before", "weight_unnormalised", "weight_after"]
        )
        writer.writerows(csv_rows)


def cnn_output_size(input_size, filter_size, padding, stride):
    size = (input_size - filter_size + 2 * padding) / stride + 1
    rounded_size = round(size)
    if not math.isclose(size, rounded_size):
        raise ValueError(f"CNN layer output dimension is not integral: {size}.")
    return int(rounded_size)


def write_results(results):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for key, value in results.items():
            if isinstance(value, (int, np.integer)):
                formatted = str(int(value))
            elif isinstance(value, (float, np.floating)):
                formatted = f"{float(value):.4f}"
            else:
                formatted = str(value)
            f.write(f"{key} = {formatted}\n")


def run_part4():
    records = common.load_records(os.path.join(ROOT, "data", "records.csv"))
    X_all = common.get_X1_to_X7(records)
    y_all = common.get_loan_default(records)
    train_indices, test_indices = part2.stratified_split(records)
    if len(train_indices) != 560 or len(test_indices) != 240:
        raise ValueError(
            f"Task 2.1 split expected 560/240 rows, got "
            f"{len(train_indices)}/{len(test_indices)}."
        )

    X_train_raw = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test_raw = X_all[test_indices]
    y_test = y_all[test_indices]

    results = {}

    # Task 4.1: split quality on the Task 2.1 training rows.
    split_masks = [
        ("S1", X_train_raw[:, 5] == 1),
        ("S2", X_train_raw[:, 6] == 1),
        ("S3", X_train_raw[:, 2] > 0.5),
        ("S4", X_train_raw[:, 0] <= 24),
    ]
    root_entropy = binary_entropy(y_train)
    root_gini = binary_gini(y_train)
    split_results = [
        evaluate_split(name, mask, y_train) for name, mask in split_masks
    ]
    best_info_split = max(split_results, key=lambda item: item["info_gain"])
    best_gini_split = max(split_results, key=lambda item: item["gini_gain"])
    results["root_entropy"] = root_entropy
    results["root_gini"] = root_gini
    results["best_split_by_info_gain"] = best_info_split["split"]
    results["best_split_by_gini_gain"] = best_gini_split["split"]

    with open(SPLITS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "n_yes",
                "positives_yes",
                "n_no",
                "positives_no",
                "info_gain",
                "gini_gain",
            ]
        )
        for split in split_results:
            writer.writerow(
                [
                    split["split"],
                    split["n_yes"],
                    split["positives_yes"],
                    split["n_no"],
                    split["positives_no"],
                    f"{split['info_gain']:.4f}",
                    f"{split['gini_gain']:.4f}",
                ]
            )

    # Task 4.2: refit the Part 2 logistic model and chosen-K KNN without
    # invoking run_part2(), which would overwrite its output files.
    X_train_std, X_test_std, _, _ = common.standardise_train_test(
        X_train_raw, X_test_raw
    )
    logistic_weights, logistic_bias = fit_logistic_regression(
        X_train_std, y_train
    )
    logistic_prob = common.sigmoid(
        X_test_std @ logistic_weights + logistic_bias
    )
    logistic_pred = (logistic_prob >= 0.3).astype(int)

    best_k = select_knn_k(X_train_std, y_train)
    knn_prob, knn_pred = knn_test_probabilities(
        X_train_std, y_train, X_test_std, best_k
    )

    stump_mask = best_info_split["mask"]
    stump_rate_yes = float(np.mean(y_train[stump_mask]))
    stump_rate_no = float(np.mean(y_train[~stump_mask]))
    results["stump_rate_yes"] = stump_rate_yes
    results["stump_rate_no"] = stump_rate_no
    stump_test_mask = {
        "S1": X_test_raw[:, 5] == 1,
        "S2": X_test_raw[:, 6] == 1,
        "S3": X_test_raw[:, 2] > 0.5,
        "S4": X_test_raw[:, 0] <= 24,
    }[best_info_split["split"]]
    stump_prob = np.where(stump_test_mask, stump_rate_yes, stump_rate_no)
    stump_pred = (stump_prob >= 0.3).astype(int)

    hard_vote_pred = (
        logistic_pred.astype(int) + knn_pred.astype(int) + stump_pred.astype(int)
        >= 2
    ).astype(int)
    soft_vote_prob = (logistic_prob + knn_prob + stump_prob) / 3.0
    soft_vote_pred = (soft_vote_prob >= 0.3).astype(int)

    predictions = [
        ("logistic", logistic_pred),
        ("knn", knn_pred),
        ("stump", stump_pred),
        ("hard_vote", hard_vote_pred),
        ("soft_vote", soft_vote_pred),
    ]
    voting_metrics = {
        name: confusion_metrics(y_test, prediction)
        for name, prediction in predictions
    }
    verify_part2_consistency(voting_metrics["logistic"], voting_metrics["knn"], best_k)
    with open(VOTING_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"]
        )
        for name, _ in predictions:
            metrics = voting_metrics[name]
            writer.writerow(
                [
                    name,
                    metrics["tp"],
                    metrics["fp"],
                    metrics["tn"],
                    metrics["fn"],
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                ]
            )

    # Task 4.3: two-round AdaBoost on the supplied eight points.
    run_adaboost(results)

    # Task 4.4: probability that a training observation is out of bag.
    results["oob_fraction"] = (1.0 - 1.0 / 560.0) ** 560

    # Task 4.5: fully connected, convolutional, and recurrent parameter counts.
    mlp_hidden1 = (784 + 1) * 16
    mlp_hidden2 = (16 + 1) * 16
    mlp_output = (16 + 1) * 10
    results["mlp_hidden1"] = mlp_hidden1
    results["mlp_hidden2"] = mlp_hidden2
    results["mlp_output"] = mlp_output
    results["mlp_total"] = mlp_hidden1 + mlp_hidden2 + mlp_output

    # The 784-pixel input corresponds to a 28x28 single-channel image.
    input_height, input_width, input_channels = 28, 28, 1
    conv1_height = cnn_output_size(input_height, 5, 2, 1)
    conv1_width = cnn_output_size(input_width, 5, 2, 1)
    conv1_channels = 6
    pool1_height = cnn_output_size(conv1_height, 2, 0, 2)
    pool1_width = cnn_output_size(conv1_width, 2, 0, 2)
    pool1_channels = conv1_channels
    conv2_height = cnn_output_size(pool1_height, 5, 0, 1)
    conv2_width = cnn_output_size(pool1_width, 5, 0, 1)
    conv2_channels = 16
    pool2_height = cnn_output_size(conv2_height, 2, 0, 2)
    pool2_width = cnn_output_size(conv2_width, 2, 0, 2)
    pool2_channels = conv2_channels
    flattened_features = pool2_height * pool2_width * pool2_channels

    cnn_conv1 = (5 * 5 * input_channels + 1) * conv1_channels
    cnn_conv2 = (5 * 5 * conv1_channels + 1) * conv2_channels
    cnn_dense1 = (flattened_features + 1) * 32
    cnn_dense2 = (32 + 1) * 10
    results["cnn_conv1"] = cnn_conv1
    results["cnn_conv2"] = cnn_conv2
    results["cnn_dense1"] = cnn_dense1
    results["cnn_dense2"] = cnn_dense2
    results["cnn_total"] = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2
    for key, dimension in (
        ("cnn_input_height", input_height),
        ("cnn_input_width", input_width),
        ("cnn_input_channels", input_channels),
        ("cnn_conv1_height", conv1_height),
        ("cnn_conv1_width", conv1_width),
        ("cnn_conv1_channels", conv1_channels),
        ("cnn_pool1_height", pool1_height),
        ("cnn_pool1_width", pool1_width),
        ("cnn_pool1_channels", pool1_channels),
        ("cnn_conv2_height", conv2_height),
        ("cnn_conv2_width", conv2_width),
        ("cnn_conv2_channels", conv2_channels),
        ("cnn_pool2_height", pool2_height),
        ("cnn_pool2_width", pool2_width),
        ("cnn_pool2_channels", pool2_channels),
        ("cnn_flattened_features", flattened_features),
    ):
        results[key] = dimension

    results["fc_layer_params"] = (128 * 128 * 3 + 1) * 1024
    results["conv_layer_params"] = (3 * 3 * 3 + 1) * 64

    input_dimension = 64
    hidden_dimension = 128
    output_dimension = 16
    rnn_w_xh = input_dimension * hidden_dimension
    rnn_w_hh = hidden_dimension * hidden_dimension
    rnn_b_h = hidden_dimension
    rnn_w_hy = hidden_dimension * output_dimension
    rnn_b_y = output_dimension
    results["rnn_total"] = (
        rnn_w_xh + rnn_w_hh + rnn_b_h + rnn_w_hy + rnn_b_y
    )

    architecture_rows = [
        ("mlp_hidden1", mlp_hidden1),
        ("mlp_hidden2", mlp_hidden2),
        ("mlp_output", mlp_output),
        ("cnn_conv1", cnn_conv1),
        ("cnn_conv2", cnn_conv2),
        ("cnn_dense1", cnn_dense1),
        ("cnn_dense2", cnn_dense2),
    ]
    with open(ARCHITECTURE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        writer.writerows(architecture_rows)

    write_results(results)
    return results


def main():
    results = run_part4()
    print("Part 4 executed successfully.")
    print(f"root_entropy = {results['root_entropy']:.4f}")
    print(f"round1_error = {results['round1_error']:.4f}")
    print(f"best_split_by_info_gain = {results['best_split_by_info_gain']}")
    print(f"cnn_total = {results['cnn_total']}")


if __name__ == "__main__":
    main()
