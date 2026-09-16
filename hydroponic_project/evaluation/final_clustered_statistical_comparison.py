#!/usr/bin/env python3

# =============================================================================
# FINAL CLUSTERED STATISTICAL COMPARISON
# IMAGE-ONLY vs CLASSICAL FUSION
# =============================================================================
#
# PURPOSE
# -------
# Provide dissertation-ready uncertainty estimates and paired statistical
# comparisons between:
#
#   E1 - New MobileNetV2 image-only
#   E3a - Classical 60/40 late fusion
#   E3b - Best exploratory late fusion from the weight scan
#
# WHY CLUSTER BY CAPTURE?
# -----------------------
# Sensor values are shared across all plants from the same capture. Therefore,
# plant observations within a capture are correlated and must not be treated as
# fully independent.
#
# This script uses two complementary statistical procedures:
#
# 1. Cluster bootstrap (95% percentile confidence interval)
#    - Resamples complete capture IDs with replacement.
#    - Estimates uncertainty of metric differences.
#
# 2. Paired cluster permutation test
#    - Randomly swaps the two model predictions WITHIN ENTIRE CAPTURES.
#    - Tests the null hypothesis that the two systems perform equivalently.
#
# METRICS
# -------
#   - Accuracy
#   - Balanced Accuracy
#   - Macro-F1
#   - Unhealthy Recall
#   - Unhealthy F1
#
# IMPORTANT
# ---------
# The "best exploratory fusion" weight was selected using the same labelled
# dataset. Therefore its comparison is optimistic and must be described as
# exploratory, not independently validated.
#
# INPUT
# -----
# /hydroponic_project/evaluation/final_image_sensor_fusion_comparison/
# final_observation_predictions.csv
#
# OUTPUT
# ------
# /hydroponic_project/evaluation/final_statistical_comparison/
#
#   clustered_metric_comparison.csv
#   cluster_bootstrap_distributions.csv
#   cluster_permutation_results.csv
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)


# =============================================================================
# 2. PATHS
# =============================================================================

INPUT_CSV = Path(
    "/hydroponic_project/evaluation/"
    "final_image_sensor_fusion_comparison/"
    "final_observation_predictions.csv"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/"
    "final_statistical_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 3. COLUMNS
# =============================================================================

GROUND_TRUTH_COLUMN = "ground_truth"
CAPTURE_COLUMN = "capture_id"

IMAGE_COLUMN = "E1_image_prediction"
FUSION_60_40_COLUMN = "E3_fusion_60_40_prediction"
BEST_FUSION_COLUMN = "E3_best_exploratory_prediction"


# =============================================================================
# 4. SETTINGS
# =============================================================================

RANDOM_SEED = 42

BOOTSTRAP_ITERATIONS = 10000

PERMUTATION_ITERATIONS = 10000


# =============================================================================
# 5. LOAD DATA
# =============================================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

required_columns = {
    GROUND_TRUTH_COLUMN,
    CAPTURE_COLUMN,
    IMAGE_COLUMN,
    FUSION_60_40_COLUMN,
    BEST_FUSION_COLUMN,
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        f"{sorted(missing_columns)}"
    )


# =============================================================================
# 6. NORMALISE LABELS
# =============================================================================

for column in [
    GROUND_TRUTH_COLUMN,
    IMAGE_COLUMN,
    FUSION_60_40_COLUMN,
    BEST_FUSION_COLUMN,
]:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

df = (
    df[
        df[GROUND_TRUTH_COLUMN].isin(
            ["healthy", "unhealthy"]
        )
    ]
    .copy()
    .reset_index(drop=True)
)

if df.empty:
    raise RuntimeError(
        "No binary labelled observations found."
    )


# =============================================================================
# 7. METRIC FUNCTION
# =============================================================================

def compute_metrics(
    y_true,
    y_pred,
):
    """
    Return all five dissertation metrics.

    Unhealthy is treated as the positive class for recall/F1.
    """

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "unhealthy_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
                    zero_division=0,
                )
            ),

        "unhealthy_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
                    zero_division=0,
                )
            ),
    }


# =============================================================================
# 8. BASELINE OBSERVED DIFFERENCE
# =============================================================================

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "unhealthy_recall",
    "unhealthy_f1",
]


def observed_metric_difference(
    dataframe,
    model_a_column,
    model_b_column,
):
    """
    Difference = Model B - Model A

    Example:
        Fusion - Image
    """

    y_true = dataframe[
        GROUND_TRUTH_COLUMN
    ].to_numpy()

    metrics_a = compute_metrics(
        y_true,
        dataframe[
            model_a_column
        ].to_numpy(),
    )

    metrics_b = compute_metrics(
        y_true,
        dataframe[
            model_b_column
        ].to_numpy(),
    )

    return {
        metric:
            metrics_b[metric]
            - metrics_a[metric]
        for metric
        in METRIC_NAMES
    }


# =============================================================================
# 9. PREPARE CAPTURE CLUSTERS
# =============================================================================

capture_ids = (
    df[
        CAPTURE_COLUMN
    ]
    .drop_duplicates()
    .to_numpy()
)

groups = {
    capture_id:
        df[
            df[
                CAPTURE_COLUMN
            ]
            == capture_id
        ].copy()
    for capture_id
    in capture_ids
}

number_of_clusters = len(
    capture_ids
)


# =============================================================================
# 10. CLUSTER BOOTSTRAP
# =============================================================================

def cluster_bootstrap(
    model_a_column,
    model_b_column,
    comparison_name,
):
    """
    Resample complete captures WITH replacement.

    For each bootstrap sample:
        difference = Model B metric - Model A metric
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):
        sampled_capture_ids = rng.choice(
            capture_ids,
            size=number_of_clusters,
            replace=True,
        )

        sample = pd.concat(
            [
                groups[
                    capture_id
                ]
                for capture_id
                in sampled_capture_ids
            ],
            ignore_index=True,
        )

        difference = (
            observed_metric_difference(
                sample,
                model_a_column,
                model_b_column,
            )
        )

        for metric, value in difference.items():
            rows.append(
                {
                    "comparison":
                        comparison_name,

                    "iteration":
                        iteration,

                    "metric":
                        metric,

                    "difference":
                        value,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 11. PAIRED CLUSTER PERMUTATION TEST
# =============================================================================

def cluster_permutation_test(
    model_a_column,
    model_b_column,
    comparison_name,
):
    """
    Null hypothesis:
        Model A and Model B are exchangeable within each capture.

    For each random permutation:
      - each capture independently gets a 50/50 swap decision
      - if swapped, ALL plant predictions for that capture are exchanged
        between Model A and Model B
      - recompute B - A metric difference

    Two-sided p-value:
      fraction of |permuted difference| >= |observed difference|

    +1 correction is applied to avoid p = 0.
    """

    observed = observed_metric_difference(
        df,
        model_a_column,
        model_b_column,
    )

    rng = np.random.default_rng(
        RANDOM_SEED + 100
    )

    permutation_distributions = {
        metric: []
        for metric
        in METRIC_NAMES
    }

    for _ in range(
        PERMUTATION_ITERATIONS
    ):

        permuted_parts = []

        for capture_id in capture_ids:

            group = groups[
                capture_id
            ].copy()

            should_swap = bool(
                rng.integers(
                    0,
                    2,
                )
            )

            if should_swap:

                temporary = (
                    group[
                        model_a_column
                    ].copy()
                )

                group[
                    model_a_column
                ] = (
                    group[
                        model_b_column
                    ].to_numpy()
                )

                group[
                    model_b_column
                ] = (
                    temporary.to_numpy()
                )

            permuted_parts.append(
                group
            )

        permuted = pd.concat(
            permuted_parts,
            ignore_index=True,
        )

        difference = (
            observed_metric_difference(
                permuted,
                model_a_column,
                model_b_column,
            )
        )

        for metric, value in difference.items():
            permutation_distributions[
                metric
            ].append(
                value
            )

    results = []

    for metric in METRIC_NAMES:

        distribution = np.asarray(
            permutation_distributions[
                metric
            ],
            dtype=float,
        )

        observed_value = float(
            observed[
                metric
            ]
        )

        extreme_count = int(
            np.sum(
                np.abs(
                    distribution
                )
                >=
                abs(
                    observed_value
                )
            )
        )

        p_value = (
            extreme_count + 1
        ) / (
            PERMUTATION_ITERATIONS + 1
        )

        results.append(
            {
                "comparison":
                    comparison_name,

                "metric":
                    metric,

                "observed_difference":
                    observed_value,

                "permutation_p_value_two_sided":
                    float(
                        p_value
                    ),

                "permutation_iterations":
                    PERMUTATION_ITERATIONS,

                "capture_clusters":
                    number_of_clusters,
            }
        )

    return pd.DataFrame(
        results
    )


# =============================================================================
# 12. RUN BOTH COMPARISONS
# =============================================================================

comparisons = [
    (
        "Fusion_60_40_minus_Image_Only",
        IMAGE_COLUMN,
        FUSION_60_40_COLUMN,
    ),
    (
        "Best_Exploratory_Fusion_minus_Image_Only",
        IMAGE_COLUMN,
        BEST_FUSION_COLUMN,
    ),
]


all_bootstrap = []

all_permutation = []

summary_rows = []


for (
    comparison_name,
    model_a_column,
    model_b_column,
) in comparisons:

    print()
    print("=" * 95)
    print(
        comparison_name
    )
    print("=" * 95)

    observed = (
        observed_metric_difference(
            df,
            model_a_column,
            model_b_column,
        )
    )

    bootstrap_df = cluster_bootstrap(
        model_a_column,
        model_b_column,
        comparison_name,
    )

    permutation_df = (
        cluster_permutation_test(
            model_a_column,
            model_b_column,
            comparison_name,
        )
    )

    all_bootstrap.append(
        bootstrap_df
    )

    all_permutation.append(
        permutation_df
    )

    for metric in METRIC_NAMES:

        metric_distribution = (
            bootstrap_df[
                bootstrap_df[
                    "metric"
                ]
                == metric
            ][
                "difference"
            ]
            .to_numpy(
                dtype=float
            )
        )

        lower = float(
            np.quantile(
                metric_distribution,
                0.025,
            )
        )

        upper = float(
            np.quantile(
                metric_distribution,
                0.975,
            )
        )

        mean_bootstrap = float(
            metric_distribution.mean()
        )

        p_value = float(
            permutation_df.loc[
                permutation_df[
                    "metric"
                ]
                == metric,
                "permutation_p_value_two_sided",
            ].iloc[
                0
            ]
        )

        summary_rows.append(
            {
                "comparison":
                    comparison_name,

                "metric":
                    metric,

                "observed_difference":
                    float(
                        observed[
                            metric
                        ]
                    ),

                "bootstrap_mean_difference":
                    mean_bootstrap,

                "bootstrap_95_ci_lower":
                    lower,

                "bootstrap_95_ci_upper":
                    upper,

                "permutation_p_value_two_sided":
                    p_value,

                "capture_clusters":
                    number_of_clusters,

                "bootstrap_iterations":
                    BOOTSTRAP_ITERATIONS,

                "permutation_iterations":
                    PERMUTATION_ITERATIONS,
            }
        )

        print(
            f"{metric:<20} "
            f"Observed = {observed[metric]:+.4f} | "
            f"95% CI [{lower:+.4f}, {upper:+.4f}] | "
            f"Permutation p = {p_value:.4f}"
        )


# =============================================================================
# 13. SAVE OUTPUTS
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)

bootstrap_distribution_df = pd.concat(
    all_bootstrap,
    ignore_index=True,
)

permutation_results_df = pd.concat(
    all_permutation,
    ignore_index=True,
)

summary_df.to_csv(
    OUTPUT_DIR
    / "clustered_metric_comparison.csv",
    index=False,
)

bootstrap_distribution_df.to_csv(
    OUTPUT_DIR
    / "cluster_bootstrap_distributions.csv",
    index=False,
)

permutation_results_df.to_csv(
    OUTPUT_DIR
    / "cluster_permutation_results.csv",
    index=False,
)


# =============================================================================
# 14. PRINT FINAL TABLE
# =============================================================================

print()
print("=" * 110)
print(
    " FINAL CLUSTERED STATISTICAL COMPARISON"
)
print("=" * 110)

print(
    "Plant observations :",
    len(
        df
    )
)

print(
    "Capture clusters   :",
    number_of_clusters
)

print(
    "Bootstrap runs     :",
    BOOTSTRAP_ITERATIONS
)

print(
    "Permutation runs   :",
    PERMUTATION_ITERATIONS
)

print()

display_columns = [
    "comparison",
    "metric",
    "observed_difference",
    "bootstrap_95_ci_lower",
    "bootstrap_95_ci_upper",
    "permutation_p_value_two_sided",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False
    )
)

print()
print(
    "INTERPRETATION RULE:"
)

print(
    "Positive difference = fusion performed better than image-only."
)

print(
    "Negative difference = fusion performed worse than image-only."
)

print(
    "If the 95% cluster-bootstrap CI includes 0 and the permutation "
    "p-value is >= 0.05, the observed difference is not statistically "
    "supported at the conventional 5% level."
)

print()
print(
    "IMPORTANT:"
)

print(
    "The best exploratory fusion was selected from these same labelled "
    "observations, so its statistical comparison is exploratory and "
    "should not be described as independent validation."
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print("=" * 110)
