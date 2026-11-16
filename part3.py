"""
Part 3: Clustering, hierarchical clustering, and PCA.
"""

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import common

RESULTS_PATH = os.path.join(ROOT, "outputs", "part3_results.txt")
SEGMENTS_PATH = os.path.join(ROOT, "outputs", "part3_segments.csv")
HIERARCHICAL_PATH = os.path.join(ROOT, "outputs", "part3_hierarchical.csv")

HIERARCHICAL_DISTANCE = np.array(
    [
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ],
    dtype=float,
)


def standardise_features(X):
    """Standardise columns using the sample standard deviation."""
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0, ddof=1)
    if np.any(stds == 0):
        raise ValueError("Cannot standardise a constant predictor column.")
    return (X - means) / stds


def run_kmeans(X, k):
    """Run the specified deterministic K-means procedure."""
    n_rows = X.shape[0]
    step = n_rows // k
    initial_indices = np.arange(k) * step
    centroids = X[initial_indices].copy()
    labels = np.full(n_rows, -1, dtype=int)

    while True:
        squared_distances = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        new_labels = np.argmin(squared_distances, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels
        for cluster_id in range(k):
            members = X[labels == cluster_id]
            if len(members) > 0:
                centroids[cluster_id] = np.mean(members, axis=0)

    residuals = X - centroids[labels]
    wcss = float(np.sum(residuals**2))
    return labels, centroids, wcss


def linkage_distance(cluster_a, cluster_b, distance_matrix, linkage):
    """Calculate a linkage distance between two clusters of matrix indices."""
    pairwise = distance_matrix[np.ix_(cluster_a, cluster_b)]
    if linkage == "single":
        return float(np.min(pairwise))
    if linkage == "complete":
        return float(np.max(pairwise))
    if linkage == "average":
        return float(np.mean(pairwise))
    raise ValueError(f"Unknown linkage method: {linkage}")


def hierarchical_merges(linkage):
    """Return five agglomerative merges for the supplied cohort distances."""
    clusters = [[i] for i in range(HIERARCHICAL_DISTANCE.shape[0])]
    merges = []

    for merge_no in range(1, HIERARCHICAL_DISTANCE.shape[0]):
        best_pair = None
        best_height = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                height = linkage_distance(
                    clusters[i], clusters[j], HIERARCHICAL_DISTANCE, linkage
                )
                if height < best_height:
                    best_height = height
                    best_pair = (i, j)

        i, j = best_pair
        first, second = sorted(
            (clusters[i], clusters[j]), key=lambda members: min(members)
        )
        merges.append((merge_no, first.copy(), second.copy(), best_height))
        combined = sorted(first + second)
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]
        clusters.append(combined)

    return merges


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


def run_part3():
    records = common.load_records(os.path.join(ROOT, "data", "records.csv"))
    X = common.get_X1_to_X5(records)
    X_std = standardise_features(X)
    y = common.get_loan_default(records)
    values = common.get_monthly_spend(records)
    n_rows = len(records)

    results = {}

    # Tasks 3.1 and 3.2: K-means WCSS and elbow selection.
    kmeans_results = {}
    for k in range(2, 7):
        labels, centroids, wcss = run_kmeans(X_std, k)
        kmeans_results[k] = (labels, centroids, wcss)
        results[f"wcss_k{k}"] = wcss

    elbow_k = next(
        (
            k
            for k in range(3, 6)
            if kmeans_results[k][2] - kmeans_results[k + 1][2]
            < 0.5 * (kmeans_results[k - 1][2] - kmeans_results[k][2])
        ),
        None,
    )
    if elbow_k is None:
        raise RuntimeError("No elbow K satisfied the specified WCSS condition.")
    results["elbow_k"] = elbow_k

    # Task 3.3: profile K=3 clusters and order them by original-unit mean X3.
    labels_k3 = kmeans_results[3][0]
    cluster_profiles = []
    for cluster_id in range(3):
        member_mask = labels_k3 == cluster_id
        cluster_profiles.append(
            {
                "cluster_id": cluster_id,
                "mask": member_mask,
                "size": int(np.sum(member_mask)),
                "mean_X3": float(np.mean(X[member_mask, 2])),
            }
        )
    cluster_profiles.sort(key=lambda profile: (profile["mean_X3"], profile["cluster_id"]))

    os.makedirs(os.path.dirname(SEGMENTS_PATH), exist_ok=True)
    with open(SEGMENTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment",
                "size",
                "mean_X1",
                "mean_X2",
                "mean_X3",
                "mean_X4",
                "mean_X5",
                "outcome_rate",
                "mean_value",
            ]
        )
        for segment_index, profile in enumerate(cluster_profiles):
            segment = chr(ord("A") + segment_index)
            member_mask = profile["mask"]
            size = profile["size"]
            results[f"size_segment_{segment}"] = size
            means = np.mean(X[member_mask], axis=0)
            outcome_rate = float(np.mean(y[member_mask]))
            mean_value = float(np.mean(values[member_mask]))
            writer.writerow(
                [
                    segment,
                    size,
                    *(f"{value:.4f}" for value in means),
                    f"{outcome_rate:.4f}",
                    f"{mean_value:.4f}",
                ]
            )

    # Task 3.4: agglomerative clustering on the supplied six-cohort matrix.
    hierarchical_results = {}
    os.makedirs(os.path.dirname(HIERARCHICAL_PATH), exist_ok=True)
    with open(HIERARCHICAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        for linkage in ("single", "complete", "average"):
            merges = hierarchical_merges(linkage)
            for merge_no, first, second, height in merges:
                first_name = "+".join(f"E{index + 1}" for index in first)
                second_name = "+".join(f"E{index + 1}" for index in second)
                writer.writerow(
                    [linkage, merge_no, first_name, second_name, f"{height:.4f}"]
                )

            heights = np.array([merge[3] for merge in merges], dtype=float)
            gaps = np.diff(heights)
            largest_gap_index = int(np.argmax(gaps))
            hierarchical_results[f"{linkage}_clusters_at_cut"] = (
                HIERARCHICAL_DISTANCE.shape[0] - (largest_gap_index + 1)
            )
    results.update(hierarchical_results)

    # Task 3.5: PCA using eigendecomposition of the sample covariance matrix.
    pca_mean = np.mean(X_std, axis=0)
    X_centered = X_std - pca_mean
    covariance = (X_centered.T @ X_centered) / (n_rows - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for component in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, component])))
        if eigenvectors[pivot, component] < 0:
            eigenvectors[:, component] *= -1.0

    explained_ratios = eigenvalues / np.sum(eigenvalues)
    cumulative_ratios = np.cumsum(explained_ratios)
    scores = X_centered @ eigenvectors[:, :2]

    results["eigenvalue_sum"] = float(np.sum(eigenvalues))
    for component in range(5):
        results[f"eigenvalue_{component + 1}"] = float(eigenvalues[component])
        results[f"explained_ratio_{component + 1}"] = float(
            explained_ratios[component]
        )
        results[f"cumulative_ratio_{component + 1}"] = float(
            cumulative_ratios[component]
        )
    results["components_for_80pct"] = int(
        np.searchsorted(cumulative_ratios, 0.80) + 1
    )

    for feature in range(5):
        results[f"pc1_loading_X{feature + 1}"] = float(eigenvectors[feature, 0])
        results[f"pc2_loading_X{feature + 1}"] = float(eigenvectors[feature, 1])
    for row in range(3):
        results[f"row{row + 1}_pc1"] = float(scores[row, 0])
        results[f"row{row + 1}_pc2"] = float(scores[row, 1])
    for segment_index, profile in enumerate(cluster_profiles):
        segment = chr(ord("A") + segment_index)
        results[f"mean_pc1_segment_{segment}"] = float(
            np.mean(scores[profile["mask"], 0])
        )

    write_results(results)
    return results


def main():
    results = run_part3()
    print("Part 3 executed successfully.")
    print(f"wcss_k2 = {results['wcss_k2']:.4f}")
    print(f"eigenvalue_sum = {results['eigenvalue_sum']:.4f}")
    print(
        "segment sizes = "
        f"{results['size_segment_A']}, "
        f"{results['size_segment_B']}, "
        f"{results['size_segment_C']}"
    )


if __name__ == "__main__":
    main()
