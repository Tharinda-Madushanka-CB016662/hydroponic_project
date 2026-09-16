#!/usr/bin/env python3

# =============================================================================
# FINAL EXPLORATORY COMPARISON
# NEW MOBILENETV2 vs NEW 5-FEATURE SENSOR RF vs CLASSICAL LATE FUSION
# + COMPLEMENTARY FOUR-STATE DECISION FUSION
# =============================================================================
#
# PURPOSE
# -------
# This script uses the SAME manually-labelled live Raspberry Pi observations
# to compare:
#
#   E1 - New MobileNetV2 image-only
#   E2 - New 5-feature Random Forest sensor-only
#        (SUPPLEMENTARY comparison against visible labels only)
#   E3 - Classical probability-level late fusion
#   E4 - Complementary decision-level fusion (four operational states)
#
# IMPORTANT CONCEPTUAL NOTE
# -------------------------
# The image and sensor branches do NOT represent exactly the same phenomenon:
#
#   Image branch:
#       current visible condition of an individual lettuce plant.
#
#   Sensor branch:
#       current shared environmental suitability / stress risk.
#
# Therefore:
#
#   Image Healthy + Sensor Stress
#       can be a valid EARLY ENVIRONMENTAL WARNING.
#
#   Image Unhealthy + Sensor Healthy
#       can be a valid VISIBLE STRESS / RESIDUAL STRESS state while current
#       environmental conditions have returned to an acceptable range.
#
# For this reason, E2 against visual labels and E3 classical late fusion are
# research experiments. E4 preserves both modalities instead of forcing them
# into a single binary meaning.
#
# DATA SOURCE
# -----------
# The previous sensor comparison CSV already contains BOTH:
#   - New MobileNetV2 probabilities
#   - New sensor RF probabilities
#   - Manual ground truth
#
# File:
# /hydroponic_project/evaluation/sensor_live_ground_truth_comparison/
# sensor_live_predictions.csv
#
# MANUAL LABELS
# -------------
# Expected binary rows:
#   Healthy   = 95
#   Unhealthy = 54
#   Total     = 149
#   Captures  = 18
#
# THRESHOLDS
# ----------
# Image threshold:
#   0.42
#   This was the earlier exploratory candidate selected from the live-labelled
#   observations.
#
# Sensor threshold:
#   0.48
#   This was selected using the sensor model's separate chronological
#   validation data.
#
# Classical fusion:
#   fusion_risk = image_weight * image_unhealthy_probability
#               + sensor_weight * sensor_unhealthy_probability
#
#   sensor_weight = 1 - image_weight
#
#   Fusion decision threshold remains fixed at 0.50 for the main sensitivity
#   analysis. We DO NOT retune the fusion decision threshold on these same
#   149 observations.
#
# FUSION WEIGHT SCAN
# ------------------
# Image weights:
#   0.00, 0.01, ... 0.99, 1.00
#
# IMPORTANT:
# The "best" fusion weight from this scan is exploratory because the SAME
# manually-labelled dataset is used to inspect the weight sensitivity.
# It must NOT be described as an independently validated production weight.
#
# OUTPUTS
# -------
# /hydroponic_project/evaluation/final_image_sensor_fusion_comparison/
#
#   final_observation_predictions.csv
#   baseline_metrics.csv
#   fusion_weight_sensitivity.csv
#   fusion_candidate_metrics.csv
#   complementary_four_state_observations.csv
#   complementary_four_state_summary.csv
#   capture_level_four_state_summary.csv
#   cluster_bootstrap_accuracy_difference.csv
#   fusion_weight_metrics.png
#   confusion_image_only.png
#   confusion_sensor_only_visual_reference.png
#   confusion_fusion_60_40.png
#   confusion_best_exploratory_fusion.png
#
# =============================================================================


# =============================================================================
# 1. IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# =============================================================================
# 2. PATHS
# =============================================================================

INPUT_CSV = Path(
    "/hydroponic_project/evaluation/"
    "sensor_live_ground_truth_comparison/"
    "sensor_live_predictions.csv"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/"
    "final_image_sensor_fusion_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 3. EXACT COLUMN NAMES FROM YOUR CURRENT CSV
# =============================================================================

GROUND_TRUTH_COLUMN = "ground_truth"

CAPTURE_COLUMN = "capture_id"

PLANT_COLUMN = "plant_id"

IMAGE_PROBABILITY_COLUMN = (
    "New_MobileNetV2_unhealthy_probability"
)

SENSOR_PROBABILITY_COLUMN = (
    "new_sensor_unhealthy_probability"
)


# =============================================================================
# 4. DECISION THRESHOLDS
# =============================================================================

IMAGE_THRESHOLD = 0.42

SENSOR_THRESHOLD = 0.48

# Main classical-fusion comparison threshold.
FUSION_THRESHOLD = 0.50


# =============================================================================
# 5. FUSION WEIGHTS TO SCAN
# =============================================================================

IMAGE_WEIGHTS = np.round(
    np.arange(
        0.00,
        1.001,
        0.01,
    ),
    2,
)


# =============================================================================
# 6. REPRODUCIBILITY FOR CLUSTER BOOTSTRAP
# =============================================================================

BOOTSTRAP_SEED = 42

BOOTSTRAP_ITERATIONS = 5000


# =============================================================================
# 7. LOAD AND VALIDATE INPUT
# =============================================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(
    INPUT_CSV
)

required_columns = {
    GROUND_TRUTH_COLUMN,
    CAPTURE_COLUMN,
    PLANT_COLUMN,
    IMAGE_PROBABILITY_COLUMN,
    SENSOR_PROBABILITY_COLUMN,
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
# 8. KEEP ONLY BINARY MANUAL LABELS
# =============================================================================

df[
    GROUND_TRUTH_COLUMN
] = (
    df[
        GROUND_TRUTH_COLUMN
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = (
    df[
        df[
            GROUND_TRUTH_COLUMN
        ].isin(
            [
                "healthy",
                "unhealthy",
            ]
        )
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

df[
    IMAGE_PROBABILITY_COLUMN
] = pd.to_numeric(
    df[
        IMAGE_PROBABILITY_COLUMN
    ],
    errors="coerce",
)

df[
    SENSOR_PROBABILITY_COLUMN
] = pd.to_numeric(
    df[
        SENSOR_PROBABILITY_COLUMN
    ],
    errors="coerce",
)

before_probability_drop = len(
    df
)

df = df.dropna(
    subset=[
        IMAGE_PROBABILITY_COLUMN,
        SENSOR_PROBABILITY_COLUMN,
    ]
).copy()

dropped_probability_rows = (
    before_probability_drop
    - len(
        df
    )
)

if df.empty:
    raise RuntimeError(
        "No valid binary observations remain."
    )


# =============================================================================
# 9. HELPER - BINARY PREDICTION FROM UNHEALTHY RISK
# =============================================================================

def risk_to_label(
    unhealthy_probability,
    threshold,
):
    return np.where(
        np.asarray(
            unhealthy_probability,
            dtype=float,
        )
        >= float(
            threshold
        ),
        "unhealthy",
        "healthy",
    )


# =============================================================================
# 10. HELPER - METRICS
# =============================================================================

def evaluate_binary(
    experiment,
    y_true,
    unhealthy_probability,
    threshold,
):
    """
    Evaluate a probability score against the MANUAL VISIBLE binary labels.

    WARNING:
    For the sensor branch, this is a supplementary cross-target comparison
    because the sensor model was trained for environmental suitability rather
    than direct visible-plant diagnosis.
    """

    unhealthy_probability = np.asarray(
        unhealthy_probability,
        dtype=float,
    )

    y_pred = risk_to_label(
        unhealthy_probability,
        threshold,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "healthy",
            "unhealthy",
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    y_binary = np.where(
        np.asarray(
            y_true
        )
        == "unhealthy",
        1,
        0,
    )

    return {
        "experiment":
            experiment,

        "threshold":
            float(
                threshold
            ),

        "samples":
            int(
                len(
                    y_true
                )
            ),

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

        "unhealthy_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
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

        "roc_auc":
            float(
                roc_auc_score(
                    y_binary,
                    unhealthy_probability,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    y_binary,
                    unhealthy_probability,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_binary,
                    unhealthy_probability,
                )
            ),

        "healthy_correct_tn":
            int(
                tn
            ),

        "healthy_as_unhealthy_fp":
            int(
                fp
            ),

        "unhealthy_as_healthy_fn":
            int(
                fn
            ),

        "unhealthy_correct_tp":
            int(
                tp
            ),
    }


# =============================================================================
# 11. HELPER - CONFUSION MATRIX FIGURE
# =============================================================================

def save_confusion_figure(
    y_true,
    y_pred,
    title,
    output_path,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "healthy",
            "unhealthy",
        ],
    )

    fig, ax = plt.subplots(
        figsize=(
            6,
            5,
        )
    )

    image = ax.imshow(
        cm
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Manual visible ground truth"
    )

    ax.set_xticks(
        [
            0,
            1,
        ]
    )

    ax.set_yticks(
        [
            0,
            1,
        ]
    )

    ax.set_xticklabels(
        [
            "Healthy",
            "Unhealthy",
        ]
    )

    ax.set_yticklabels(
        [
            "Healthy",
            "Unhealthy",
        ]
    )

    for row in range(
        cm.shape[
            0
        ]
    ):
        for column in range(
            cm.shape[
                1
            ]
        ):
            ax.text(
                column,
                row,
                int(
                    cm[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# 12. E1 - IMAGE ONLY
# =============================================================================

y_true = df[
    GROUND_TRUTH_COLUMN
].to_numpy()

image_probability = df[
    IMAGE_PROBABILITY_COLUMN
].to_numpy(
    dtype=float
)

sensor_probability = df[
    SENSOR_PROBABILITY_COLUMN
].to_numpy(
    dtype=float
)


image_metrics = evaluate_binary(
    experiment=
        "E1_New_MobileNetV2_Image_Only",

    y_true=
        y_true,

    unhealthy_probability=
        image_probability,

    threshold=
        IMAGE_THRESHOLD,
)

image_prediction = risk_to_label(
    image_probability,
    IMAGE_THRESHOLD,
)


# =============================================================================
# 13. E2 - SENSOR ONLY AGAINST MANUAL VISIBLE LABELS
# =============================================================================
#
# IMPORTANT:
# This is NOT the primary accuracy result of the sensor model.
#
# The primary sensor-model evaluation is against expert-derived environmental
# labels. This E2 result is included only to quantify same-time agreement with
# visible plant condition.
# =============================================================================

sensor_metrics = evaluate_binary(
    experiment=
        "E2_New_RF_5Feature_Sensor_Only_VisualReference",

    y_true=
        y_true,

    unhealthy_probability=
        sensor_probability,

    threshold=
        SENSOR_THRESHOLD,
)

sensor_prediction = risk_to_label(
    sensor_probability,
    SENSOR_THRESHOLD,
)


# =============================================================================
# 14. E3 - CLASSICAL LATE-FUSION WEIGHT SENSITIVITY
# =============================================================================

fusion_rows = []

for image_weight in IMAGE_WEIGHTS:

    sensor_weight = round(
        1.0
        - float(
            image_weight
        ),
        2,
    )

    fusion_probability = (
        float(
            image_weight
        )
        * image_probability
        +
        sensor_weight
        * sensor_probability
    )

    metrics = evaluate_binary(
        experiment=
            "E3_Classical_Late_Fusion",

        y_true=
            y_true,

        unhealthy_probability=
            fusion_probability,

        threshold=
            FUSION_THRESHOLD,
    )

    metrics[
        "image_weight"
    ] = float(
        image_weight
    )

    metrics[
        "sensor_weight"
    ] = float(
        sensor_weight
    )

    fusion_rows.append(
        metrics
    )


fusion_df = pd.DataFrame(
    fusion_rows
)


# =============================================================================
# 15. SELECT EXPLORATORY BEST FUSION WEIGHT
# =============================================================================
#
# Primary criterion:
#   Macro-F1
#
# Tie breakers:
#   Balanced Accuracy
#   Unhealthy F1
#   Unhealthy Recall
#   Higher image weight only as the final tie-break because the evaluation
#   target is visible plant condition.
#
# This does NOT mean image weight is manually forced to be high.
# =============================================================================

best_fusion_row = (
    fusion_df
    .sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy",
            "unhealthy_f1",
            "unhealthy_recall",
            "image_weight",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
        ],
    )
    .iloc[
        0
    ]
)


best_image_weight = float(
    best_fusion_row[
        "image_weight"
    ]
)

best_sensor_weight = float(
    best_fusion_row[
        "sensor_weight"
    ]
)

best_fusion_probability = (
    best_image_weight
    * image_probability
    +
    best_sensor_weight
    * sensor_probability
)

best_fusion_prediction = risk_to_label(
    best_fusion_probability,
    FUSION_THRESHOLD,
)


# =============================================================================
# 16. REFERENCE 60/40 FUSION
# =============================================================================

fusion_60_40_probability = (
    0.60
    * image_probability
    +
    0.40
    * sensor_probability
)

fusion_60_40_metrics = evaluate_binary(
    experiment=
        "E3_Reference_Fusion_Image060_Sensor040",

    y_true=
        y_true,

    unhealthy_probability=
        fusion_60_40_probability,

    threshold=
        FUSION_THRESHOLD,
)

fusion_60_40_metrics[
    "image_weight"
] = 0.60

fusion_60_40_metrics[
    "sensor_weight"
] = 0.40

fusion_60_40_prediction = risk_to_label(
    fusion_60_40_probability,
    FUSION_THRESHOLD,
)


# =============================================================================
# 17. OTHER USEFUL FIXED FUSION REFERENCES
# =============================================================================

fixed_candidates = [
    (
        "Fusion_50_50",
        0.50,
    ),
    (
        "Fusion_60_40",
        0.60,
    ),
    (
        "Fusion_70_30",
        0.70,
    ),
    (
        "Fusion_80_20",
        0.80,
    ),
    (
        "Fusion_90_10",
        0.90,
    ),
]

candidate_rows = []

for candidate_name, image_weight in fixed_candidates:

    sensor_weight = (
        1.0
        - image_weight
    )

    probability = (
        image_weight
        * image_probability
        +
        sensor_weight
        * sensor_probability
    )

    metrics = evaluate_binary(
        experiment=
            candidate_name,

        y_true=
            y_true,

        unhealthy_probability=
            probability,

        threshold=
            FUSION_THRESHOLD,
    )

    metrics[
        "image_weight"
    ] = image_weight

    metrics[
        "sensor_weight"
    ] = sensor_weight

    candidate_rows.append(
        metrics
    )


# Add the exploratory best row.
candidate_rows.append(
    {
        **best_fusion_row.to_dict(),
        "experiment":
            (
                "Best_Exploratory_Fusion_"
                f"Image{best_image_weight:.2f}_"
                f"Sensor{best_sensor_weight:.2f}"
            ),
    }
)

candidate_df = pd.DataFrame(
    candidate_rows
)


# =============================================================================
# 18. E4 - COMPLEMENTARY FOUR-STATE DECISION FUSION
# =============================================================================
#
# This is NOT another binary classifier.
#
# It preserves the meaning of both branches.
#
# A. Image Healthy + Sensor Healthy
#       HEALTHY / STABLE
#
# B. Image Healthy + Sensor Stress
#       EARLY ENVIRONMENTAL WARNING
#
# C. Image Unhealthy + Sensor Healthy
#       VISIBLE PLANT STRESS - ENVIRONMENT CURRENTLY NORMAL
#
# D. Image Unhealthy + Sensor Stress
#       HIGH RISK - VISIBLE + ENVIRONMENTAL STRESS
# =============================================================================

def complementary_state(
    image_label,
    sensor_label,
):

    if (
        image_label
        == "healthy"
        and sensor_label
        == "healthy"
    ):
        return (
            "HEALTHY / STABLE"
        )

    if (
        image_label
        == "healthy"
        and sensor_label
        == "unhealthy"
    ):
        return (
            "EARLY ENVIRONMENTAL WARNING"
        )

    if (
        image_label
        == "unhealthy"
        and sensor_label
        == "healthy"
    ):
        return (
            "VISIBLE PLANT STRESS - ENVIRONMENT CURRENTLY NORMAL"
        )

    return (
        "HIGH RISK - VISIBLE + ENVIRONMENTAL STRESS"
    )


df[
    "E1_image_prediction"
] = image_prediction

df[
    "E2_sensor_environment_prediction"
] = sensor_prediction

df[
    "E4_complementary_state"
] = [
    complementary_state(
        image_label,
        sensor_label,
    )
    for (
        image_label,
        sensor_label,
    )
    in zip(
        image_prediction,
        sensor_prediction,
    )
]


# =============================================================================
# 19. FOUR-STATE SUMMARY
# =============================================================================

four_state_summary = (
    df
    .groupby(
        [
            "E4_complementary_state",
            GROUND_TRUTH_COLUMN,
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)

for expected_column in [
    "healthy",
    "unhealthy",
]:
    if expected_column not in four_state_summary.columns:
        four_state_summary[
            expected_column
        ] = 0

four_state_summary[
    "total"
] = (
    four_state_summary[
        "healthy"
    ]
    +
    four_state_summary[
        "unhealthy"
    ]
)

four_state_summary[
    "manual_visible_unhealthy_fraction"
] = (
    four_state_summary[
        "unhealthy"
    ]
    /
    four_state_summary[
        "total"
    ]
)


# =============================================================================
# 20. CAPTURE-LEVEL FOUR-STATE SUMMARY
# =============================================================================

capture_state_summary = (
    df
    .groupby(
        CAPTURE_COLUMN
    )
    .agg(
        labelled_plants=(
            PLANT_COLUMN,
            "size",
        ),

        manual_healthy=(
            GROUND_TRUTH_COLUMN,
            lambda values:
                int(
                    (
                        values
                        == "healthy"
                    ).sum()
                ),
        ),

        manual_unhealthy=(
            GROUND_TRUTH_COLUMN,
            lambda values:
                int(
                    (
                        values
                        == "unhealthy"
                    ).sum()
                ),
        ),

        sensor_environment_prediction=(
            "E2_sensor_environment_prediction",
            "first",
        ),

        sensor_unhealthy_probability=(
            SENSOR_PROBABILITY_COLUMN,
            "first",
        ),
    )
    .reset_index()
)

capture_state_summary[
    "manual_visible_unhealthy_fraction"
] = (
    capture_state_summary[
        "manual_unhealthy"
    ]
    /
    capture_state_summary[
        "labelled_plants"
    ]
)


# =============================================================================
# 21. CLUSTER BOOTSTRAP
# =============================================================================
#
# Why cluster by capture?
# -----------------------
# Sensor probability is shared by all plants in the same capture.
# Plant observations within one capture are therefore not independent.
#
# We resample whole CAPTURES rather than individual plant rows.
#
# IMPORTANT:
# This is an exploratory uncertainty analysis only.
# The fusion weight was inspected on the same dataset, so the CI for the
# "best exploratory fusion" is optimistic and must not be called independent
# validation.
# =============================================================================

def cluster_bootstrap_accuracy_difference(
    dataframe,
    image_predictions,
    comparison_predictions,
    iterations,
    seed,
):
    work = dataframe[
        [
            CAPTURE_COLUMN,
            GROUND_TRUTH_COLUMN,
        ]
    ].copy()

    work[
        "_image_prediction"
    ] = image_predictions

    work[
        "_comparison_prediction"
    ] = comparison_predictions

    capture_ids = (
        work[
            CAPTURE_COLUMN
        ]
        .drop_duplicates()
        .to_numpy()
    )

    groups = {
        capture_id:
            work[
                work[
                    CAPTURE_COLUMN
                ]
                == capture_id
            ]
        for capture_id
        in capture_ids
    }

    rng = np.random.default_rng(
        seed
    )

    differences = []

    for _ in range(
        iterations
    ):

        sampled_capture_ids = rng.choice(
            capture_ids,
            size=len(
                capture_ids
            ),
            replace=True,
        )

        sampled_groups = [
            groups[
                capture_id
            ]
            for capture_id
            in sampled_capture_ids
        ]

        sample = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

        image_accuracy = accuracy_score(
            sample[
                GROUND_TRUTH_COLUMN
            ],
            sample[
                "_image_prediction"
            ],
        )

        comparison_accuracy = accuracy_score(
            sample[
                GROUND_TRUTH_COLUMN
            ],
            sample[
                "_comparison_prediction"
            ],
        )

        differences.append(
            comparison_accuracy
            - image_accuracy
        )

    differences = np.asarray(
        differences,
        dtype=float,
    )

    return {
        "mean_accuracy_difference":
            float(
                differences.mean()
            ),

        "ci_2_5":
            float(
                np.quantile(
                    differences,
                    0.025,
                )
            ),

        "ci_97_5":
            float(
                np.quantile(
                    differences,
                    0.975,
                )
            ),

        "bootstrap_iterations":
            int(
                iterations
            ),

        "unique_capture_clusters":
            int(
                len(
                    capture_ids
                )
            ),
    }


bootstrap_rows = []

for name, prediction in [
    (
        "Fusion_60_40_minus_Image_Only",
        fusion_60_40_prediction,
    ),
    (
        "Best_Exploratory_Fusion_minus_Image_Only",
        best_fusion_prediction,
    ),
]:

    result = cluster_bootstrap_accuracy_difference(
        dataframe=
            df,

        image_predictions=
            image_prediction,

        comparison_predictions=
            prediction,

        iterations=
            BOOTSTRAP_ITERATIONS,

        seed=
            BOOTSTRAP_SEED,
    )

    result[
        "comparison"
    ] = name

    bootstrap_rows.append(
        result
    )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# =============================================================================
# 22. SAVE OBSERVATION-LEVEL RESULTS
# =============================================================================

df[
    "E1_image_unhealthy_probability"
] = image_probability

df[
    "E1_image_threshold"
] = IMAGE_THRESHOLD

df[
    "E2_sensor_unhealthy_probability"
] = sensor_probability

df[
    "E2_sensor_threshold"
] = SENSOR_THRESHOLD

df[
    "E3_fusion_60_40_unhealthy_probability"
] = fusion_60_40_probability

df[
    "E3_fusion_60_40_prediction"
] = fusion_60_40_prediction

df[
    "E3_best_exploratory_image_weight"
] = best_image_weight

df[
    "E3_best_exploratory_sensor_weight"
] = best_sensor_weight

df[
    "E3_best_exploratory_unhealthy_probability"
] = best_fusion_probability

df[
    "E3_best_exploratory_prediction"
] = best_fusion_prediction


# =============================================================================
# 23. SAVE CSV OUTPUTS
# =============================================================================

baseline_metrics_df = pd.DataFrame(
    [
        image_metrics,
        sensor_metrics,
        fusion_60_40_metrics,
    ]
)

baseline_metrics_df.to_csv(
    OUTPUT_DIR
    / "baseline_metrics.csv",
    index=False,
)

fusion_df.to_csv(
    OUTPUT_DIR
    / "fusion_weight_sensitivity.csv",
    index=False,
)

candidate_df.to_csv(
    OUTPUT_DIR
    / "fusion_candidate_metrics.csv",
    index=False,
)

df.to_csv(
    OUTPUT_DIR
    / "final_observation_predictions.csv",
    index=False,
)

df[
    [
        CAPTURE_COLUMN,
        PLANT_COLUMN,
        GROUND_TRUTH_COLUMN,
        "E1_image_prediction",
        "E2_sensor_environment_prediction",
        "E4_complementary_state",
    ]
].to_csv(
    OUTPUT_DIR
    / "complementary_four_state_observations.csv",
    index=False,
)

four_state_summary.to_csv(
    OUTPUT_DIR
    / "complementary_four_state_summary.csv",
    index=False,
)

capture_state_summary.to_csv(
    OUTPUT_DIR
    / "capture_level_four_state_summary.csv",
    index=False,
)

bootstrap_df.to_csv(
    OUTPUT_DIR
    / "cluster_bootstrap_accuracy_difference.csv",
    index=False,
)


# =============================================================================
# 24. FUSION SENSITIVITY PLOT
# =============================================================================

plt.figure(
    figsize=(
        10,
        6,
    )
)

plt.plot(
    fusion_df[
        "image_weight"
    ],
    fusion_df[
        "accuracy"
    ],
    label="Accuracy",
)

plt.plot(
    fusion_df[
        "image_weight"
    ],
    fusion_df[
        "balanced_accuracy"
    ],
    label="Balanced Accuracy",
)

plt.plot(
    fusion_df[
        "image_weight"
    ],
    fusion_df[
        "macro_f1"
    ],
    label="Macro-F1",
)

plt.plot(
    fusion_df[
        "image_weight"
    ],
    fusion_df[
        "unhealthy_recall"
    ],
    label="Visible Unhealthy Recall",
)

plt.axvline(
    0.60,
    linestyle="--",
    label="Old 60/40 reference",
)

plt.axvline(
    best_image_weight,
    linestyle=":",
    label=(
        "Best exploratory image weight "
        f"= {best_image_weight:.2f}"
    ),
)

plt.xlabel(
    "Image weight"
)

plt.ylabel(
    "Metric value"
)

plt.title(
    "New MobileNetV2 + New Sensor RF Late-Fusion Weight Sensitivity"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "fusion_weight_metrics.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# 25. CONFUSION MATRIX FIGURES
# =============================================================================

save_confusion_figure(
    y_true=
        y_true,

    y_pred=
        image_prediction,

    title=
        "E1 - New MobileNetV2 Image Only",

    output_path=
        OUTPUT_DIR
        / "confusion_image_only.png",
)

save_confusion_figure(
    y_true=
        y_true,

    y_pred=
        sensor_prediction,

    title=
        (
            "E2 - New Sensor RF vs Same-Time "
            "Visible Labels (Supplementary)"
        ),

    output_path=
        OUTPUT_DIR
        / "confusion_sensor_only_visual_reference.png",
)

save_confusion_figure(
    y_true=
        y_true,

    y_pred=
        fusion_60_40_prediction,

    title=
        "E3 - Classical Fusion 60% Image / 40% Sensor",

    output_path=
        OUTPUT_DIR
        / "confusion_fusion_60_40.png",
)

save_confusion_figure(
    y_true=
        y_true,

    y_pred=
        best_fusion_prediction,

    title=
        (
            "E3 - Best Exploratory Fusion "
            f"{best_image_weight:.2f}/{best_sensor_weight:.2f}"
        ),

    output_path=
        OUTPUT_DIR
        / "confusion_best_exploratory_fusion.png",
)


# =============================================================================
# 26. PRINT FINAL SUMMARY
# =============================================================================

print()
print("=" * 118)
print(
    " FINAL IMAGE vs SENSOR vs CLASSICAL FUSION COMPARISON"
)
print("=" * 118)

print(
    "Binary manual observations :",
    len(
        df
    )
)

print(
    "Healthy manual labels      :",
    int(
        (
            df[
                GROUND_TRUTH_COLUMN
            ]
            == "healthy"
        ).sum()
    )
)

print(
    "Unhealthy manual labels    :",
    int(
        (
            df[
                GROUND_TRUTH_COLUMN
            ]
            == "unhealthy"
        ).sum()
    )
)

print(
    "Unique capture clusters    :",
    df[
        CAPTURE_COLUMN
    ].nunique()
)

print(
    "Dropped probability rows   :",
    dropped_probability_rows
)

print()
print(
    f"Image threshold             : {IMAGE_THRESHOLD:.2f}"
)

print(
    f"Sensor threshold            : {SENSOR_THRESHOLD:.2f}"
)

print(
    f"Fusion threshold            : {FUSION_THRESHOLD:.2f}"
)

print()
print("=" * 118)
print(
    " BASELINE / REFERENCE RESULTS AGAINST MANUAL VISIBLE LABELS"
)
print("=" * 118)

display_columns = [
    "experiment",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "unhealthy_precision",
    "unhealthy_recall",
    "unhealthy_f1",
    "roc_auc",
    "brier_score",
    "healthy_as_unhealthy_fp",
    "unhealthy_as_healthy_fn",
    "unhealthy_correct_tp",
]

print(
    baseline_metrics_df[
        display_columns
    ].to_string(
        index=False
    )
)

print()
print("=" * 118)
print(
    " EXPLORATORY BEST CLASSICAL FUSION"
)
print("=" * 118)

print(
    "Image weight      :",
    f"{best_image_weight:.2f}"
)

print(
    "Sensor weight     :",
    f"{best_sensor_weight:.2f}"
)

print(
    "Accuracy          :",
    f"{best_fusion_row['accuracy']:.4f}"
)

print(
    "Balanced Accuracy :",
    f"{best_fusion_row['balanced_accuracy']:.4f}"
)

print(
    "Macro-F1          :",
    f"{best_fusion_row['macro_f1']:.4f}"
)

print(
    "Unhealthy Recall  :",
    f"{best_fusion_row['unhealthy_recall']:.4f}"
)

print(
    "Unhealthy F1      :",
    f"{best_fusion_row['unhealthy_f1']:.4f}"
)

print()
print(
    "Image-only Macro-F1:",
    f"{image_metrics['macro_f1']:.4f}"
)

print(
    "Fusion - Image Macro-F1 difference:",
    f"{(
        best_fusion_row['macro_f1']
        - image_metrics['macro_f1']
    ):+.4f}"
)

print()
print("=" * 118)
print(
    " COMPLEMENTARY FOUR-STATE DECISION FUSION"
)
print("=" * 118)

print(
    four_state_summary.to_string(
        index=False
    )
)

print()
print(
    "INTERPRETATION:"
)

print(
    "HEALTHY / STABLE = image and environmental branch both normal."
)

print(
    "EARLY ENVIRONMENTAL WARNING = plant currently looks healthy "
    "but the environmental branch indicates stress risk."
)

print(
    "VISIBLE PLANT STRESS - ENVIRONMENT CURRENTLY NORMAL = visible "
    "stress is present while current shared sensor conditions are normal."
)

print(
    "HIGH RISK = visible stress and environmental stress are both present."
)

print()
print("=" * 118)
print(
    " CAPTURE-CLUSTER BOOTSTRAP: ACCURACY DIFFERENCE vs IMAGE ONLY"
)
print("=" * 118)

print(
    bootstrap_df[
        [
            "comparison",
            "mean_accuracy_difference",
            "ci_2_5",
            "ci_97_5",
            "bootstrap_iterations",
            "unique_capture_clusters",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "IMPORTANT RESEARCH CAUTIONS:"
)

print(
    "1. The sensor-only result against visible manual labels is a "
    "cross-target supplementary analysis, not the primary sensor accuracy."
)

print(
    "2. The fusion-weight scan is exploratory because these same manual "
    "labels are used to inspect the fusion weights."
)

print(
    "3. Sensor measurements are shared at capture/system level, so plant "
    "rows within a capture are clustered rather than independent."
)

print(
    "4. The complementary four-state decision layer should not be compared "
    "using ordinary binary accuracy because it preserves two different "
    "kinds of information instead of predicting one binary target."
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print("=" * 118)
