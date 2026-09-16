#!/usr/bin/env python3

# ============================================================
# NEW MOBILENETV2 IMAGE-THRESHOLD SENSITIVITY ANALYSIS
# ============================================================
#
# Purpose
# -------
# Evaluate how changing the decision threshold for the NEW
# stress-trained MobileNetV2 changes performance against the
# manually labelled live Raspberry Pi observations.
#
# The model already produced an "unhealthy probability" for
# every labelled observation. Therefore this script does NOT
# rerun CNN inference. It simply applies different thresholds
# to those saved probabilities.
#
# Example:
#
#     unhealthy_probability >= threshold -> unhealthy
#     unhealthy_probability <  threshold -> healthy
#
# IMPORTANT RESEARCH NOTE
# -----------------------
# This is an EXPLORATORY threshold sensitivity analysis.
#
# The same manually labelled observations are being used to
# compare thresholds, therefore the threshold with the highest
# score must NOT immediately be claimed as an independently
# validated production threshold.
#
# The selected candidate threshold should later be validated on
# additional manually labelled captures that were NOT used to
# choose it.
#
# Input
# -----
# /hydroponic_project/evaluation/live_model_evaluation/
#     live_image_model_predictions.csv
#
# Outputs
# -------
# /hydroponic_project/evaluation/threshold_sensitivity/
#
#     new_mobilenet_threshold_sensitivity.csv
#     new_mobilenet_threshold_summary.csv
#     new_mobilenet_threshold_metrics.png
#     new_mobilenet_threshold_errors.png
#
# Metrics
# -------
# - Accuracy
# - Balanced Accuracy
# - Macro-F1
# - Unhealthy Precision
# - Unhealthy Recall
# - Unhealthy F1
# - TN / FP / FN / TP
# ============================================================


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
)


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/threshold_sensitivity"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "new_mobilenet_threshold_sensitivity.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "new_mobilenet_threshold_summary.csv"
)

METRICS_PLOT = (
    OUTPUT_DIR
    / "new_mobilenet_threshold_metrics.png"
)

ERRORS_PLOT = (
    OUTPUT_DIR
    / "new_mobilenet_threshold_errors.png"
)


# ============================================================
# COLUMN NAMES
# ============================================================

GROUND_TRUTH_COLUMN = (
    "ground_truth"
)

PROBABILITY_COLUMN = (
    "New_MobileNetV2_unhealthy_probability"
)


# ============================================================
# THRESHOLD SCAN
# ============================================================
#
# 0.05 -> 0.95 in steps of 0.01
#
# This includes:
#     0.50 = current/default threshold
#
# and gives sufficiently fine resolution for sensitivity
# analysis without pretending that extremely precise values
# are biologically meaningful.
# ============================================================

THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01

DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input predictions CSV not found:\n"
            f"{INPUT_CSV}"
        )


    df = pd.read_csv(
        INPUT_CSV
    )


    required_columns = {
        GROUND_TRUTH_COLUMN,
        PROBABILITY_COLUMN,
    }


    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Required columns are missing from the CSV: "
            f"{sorted(missing_columns)}"
        )


    # --------------------------------------------------------
    # NORMALISE GROUND TRUTH
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # KEEP ONLY BINARY GROUND TRUTH
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VALIDATE PROBABILITIES
    # --------------------------------------------------------

    df[
        PROBABILITY_COLUMN
    ] = pd.to_numeric(
        df[
            PROBABILITY_COLUMN
        ],
        errors="coerce",
    )


    missing_probability = int(
        df[
            PROBABILITY_COLUMN
        ]
        .isna()
        .sum()
    )


    if missing_probability > 0:

        raise RuntimeError(
            f"{missing_probability} rows have missing or "
            "invalid New MobileNetV2 probabilities."
        )


    invalid_probability = df[
        ~df[
            PROBABILITY_COLUMN
        ].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ]


    if not invalid_probability.empty:

        raise RuntimeError(
            "Some unhealthy probabilities are outside "
            "the valid [0, 1] range."
        )


    if df.empty:

        raise RuntimeError(
            "No Healthy/Unhealthy observations are available."
        )


    return df


# ============================================================
# CALCULATE METRICS FOR ONE THRESHOLD
# ============================================================

def evaluate_threshold(
    df,
    threshold,
):

    y_true = (
        df[
            GROUND_TRUTH_COLUMN
        ]
        .to_numpy()
    )


    probabilities = (
        df[
            PROBABILITY_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_pred = np.where(
        probabilities
        >= threshold,
        "unhealthy",
        "healthy",
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "healthy",
            "unhealthy",
        ],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    return {
        "threshold":
            float(
                threshold
            ),

        "samples":
            int(
                len(
                    df
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


# ============================================================
# HELPER: SELECT BEST ROW FOR ONE METRIC
# ============================================================

def best_row(
    results_df,
    metric,
):

    # Sort by requested metric first.
    #
    # Ties are broken by:
    #     higher balanced accuracy
    #     higher unhealthy F1
    #     fewer false negatives
    #     threshold closest to 0.50
    #
    # This tie-break is for reporting only.
    # It does not make the result independently validated.

    temporary = (
        results_df
        .copy()
    )


    temporary[
        "distance_from_default"
    ] = (
        temporary[
            "threshold"
        ]
        - DEFAULT_THRESHOLD
    ).abs()


    temporary = (
        temporary
        .sort_values(
            by=[
                metric,
                "balanced_accuracy",
                "unhealthy_f1",
                "unhealthy_as_healthy_fn",
                "distance_from_default",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    return temporary.iloc[0]


# ============================================================
# PLOTS
# ============================================================

def create_metric_plot(
    results_df,
):

    plt.figure(
        figsize=(
            10,
            6,
        )
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "accuracy"
        ],
        label="Accuracy",
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "balanced_accuracy"
        ],
        label="Balanced Accuracy",
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "macro_f1"
        ],
        label="Macro-F1",
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "unhealthy_recall"
        ],
        label="Unhealthy Recall",
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "unhealthy_f1"
        ],
        label="Unhealthy F1",
    )


    plt.axvline(
        DEFAULT_THRESHOLD,
        linestyle="--",
        label="Default threshold = 0.50",
    )


    plt.xlabel(
        "Unhealthy probability threshold"
    )


    plt.ylabel(
        "Metric value"
    )


    plt.title(
        "New MobileNetV2 Live Threshold Sensitivity"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        METRICS_PLOT,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


def create_error_plot(
    results_df,
):

    plt.figure(
        figsize=(
            10,
            6,
        )
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "healthy_as_unhealthy_fp"
        ],
        label="False Positives",
    )


    plt.plot(
        results_df[
            "threshold"
        ],
        results_df[
            "unhealthy_as_healthy_fn"
        ],
        label="False Negatives",
    )


    plt.axvline(
        DEFAULT_THRESHOLD,
        linestyle="--",
        label="Default threshold = 0.50",
    )


    plt.xlabel(
        "Unhealthy probability threshold"
    )


    plt.ylabel(
        "Number of errors"
    )


    plt.title(
        "New MobileNetV2 Threshold Error Trade-off"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        ERRORS_PLOT,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = load_data()


    healthy_count = int(
        (
            df[
                GROUND_TRUTH_COLUMN
            ]
            == "healthy"
        )
        .sum()
    )


    unhealthy_count = int(
        (
            df[
                GROUND_TRUTH_COLUMN
            ]
            == "unhealthy"
        )
        .sum()
    )


    unique_captures = (
        df[
            "capture_id"
        ].nunique()
        if "capture_id"
        in df.columns
        else np.nan
    )


    print()
    print("=" * 92)
    print(
        " NEW MOBILENETV2 IMAGE-THRESHOLD "
        "SENSITIVITY ANALYSIS"
    )
    print("=" * 92)


    print(
        f"Observations      : "
        f"{len(df)}"
    )


    print(
        f"Healthy           : "
        f"{healthy_count}"
    )


    print(
        f"Unhealthy         : "
        f"{unhealthy_count}"
    )


    print(
        f"Unique captures   : "
        f"{unique_captures}"
    )


    print(
        f"Threshold range   : "
        f"{THRESHOLD_START:.2f} "
        f"to "
        f"{THRESHOLD_END:.2f}"
    )


    print(
        f"Threshold step    : "
        f"{THRESHOLD_STEP:.2f}"
    )


    print()


    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END
        + (
            THRESHOLD_STEP
            / 2.0
        ),
        THRESHOLD_STEP,
    )


    results = []


    for threshold in thresholds:

        threshold = round(
            float(
                threshold
            ),
            2,
        )


        results.append(
            evaluate_threshold(
                df,
                threshold,
            )
        )


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )


    # ========================================================
    # IMPORTANT REFERENCE ROWS
    # ========================================================

    default_row = (
        results_df[
            np.isclose(
                results_df[
                    "threshold"
                ],
                DEFAULT_THRESHOLD,
            )
        ]
        .iloc[0]
    )


    best_macro = best_row(
        results_df,
        "macro_f1",
    )


    best_balanced = best_row(
        results_df,
        "balanced_accuracy",
    )


    best_unhealthy_f1 = best_row(
        results_df,
        "unhealthy_f1",
    )


    best_accuracy = best_row(
        results_df,
        "accuracy",
    )


    summary_rows = []


    for (
        selection_name,
        row,
    ) in [
        (
            "default_0.50",
            default_row,
        ),
        (
            "best_macro_f1",
            best_macro,
        ),
        (
            "best_balanced_accuracy",
            best_balanced,
        ),
        (
            "best_unhealthy_f1",
            best_unhealthy_f1,
        ),
        (
            "best_accuracy",
            best_accuracy,
        ),
    ]:

        summary_rows.append(
            {
                "selection":
                    selection_name,

                "threshold":
                    float(
                        row[
                            "threshold"
                        ]
                    ),

                "accuracy":
                    float(
                        row[
                            "accuracy"
                        ]
                    ),

                "balanced_accuracy":
                    float(
                        row[
                            "balanced_accuracy"
                        ]
                    ),

                "macro_f1":
                    float(
                        row[
                            "macro_f1"
                        ]
                    ),

                "unhealthy_precision":
                    float(
                        row[
                            "unhealthy_precision"
                        ]
                    ),

                "unhealthy_recall":
                    float(
                        row[
                            "unhealthy_recall"
                        ]
                    ),

                "unhealthy_f1":
                    float(
                        row[
                            "unhealthy_f1"
                        ]
                    ),

                "false_positives":
                    int(
                        row[
                            "healthy_as_unhealthy_fp"
                        ]
                    ),

                "false_negatives":
                    int(
                        row[
                            "unhealthy_as_healthy_fn"
                        ]
                    ),

                "true_positives":
                    int(
                        row[
                            "unhealthy_correct_tp"
                        ]
                    ),

                "true_negatives":
                    int(
                        row[
                            "healthy_correct_tn"
                        ]
                    ),
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )


    # ========================================================
    # CREATE FIGURES
    # ========================================================

    create_metric_plot(
        results_df
    )


    create_error_plot(
        results_df
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("=" * 92)
    print(
        " THRESHOLD SUMMARY"
    )
    print("=" * 92)


    print(
        summary_df.to_string(
            index=False,
            formatters={
                "threshold":
                    lambda x:
                        f"{x:.2f}",

                "accuracy":
                    lambda x:
                        f"{x:.4f}",

                "balanced_accuracy":
                    lambda x:
                        f"{x:.4f}",

                "macro_f1":
                    lambda x:
                        f"{x:.4f}",

                "unhealthy_precision":
                    lambda x:
                        f"{x:.4f}",

                "unhealthy_recall":
                    lambda x:
                        f"{x:.4f}",

                "unhealthy_f1":
                    lambda x:
                        f"{x:.4f}",
            },
        )
    )


    print()
    print("=" * 92)
    print(
        " INTERPRETATION"
    )
    print("=" * 92)


    print(
        f"Current/default threshold : "
        f"{DEFAULT_THRESHOLD:.2f}"
    )


    print(
        f"Best exploratory Macro-F1 : "
        f"{best_macro['threshold']:.2f} "
        f"(Macro-F1 "
        f"{best_macro['macro_f1']:.4f})"
    )


    print(
        f"Best exploratory Bal. Acc : "
        f"{best_balanced['threshold']:.2f} "
        f"(Balanced Accuracy "
        f"{best_balanced['balanced_accuracy']:.4f})"
    )


    print(
        f"Best exploratory Unhealthy F1 : "
        f"{best_unhealthy_f1['threshold']:.2f} "
        f"(Unhealthy F1 "
        f"{best_unhealthy_f1['unhealthy_f1']:.4f})"
    )


    print()
    print(
        "IMPORTANT:"
    )


    print(
        "The threshold producing the highest score here is "
        "a candidate threshold, not an independently validated "
        "production threshold."
    )


    print(
        "Validate the candidate threshold on additional "
        "manually labelled captures collected after this "
        "threshold-selection analysis."
    )


    print()


    print(
        "Full sensitivity CSV:"
    )

    print(
        OUTPUT_CSV
    )


    print(
        "Summary CSV:"
    )

    print(
        SUMMARY_CSV
    )


    print(
        "Metrics plot:"
    )

    print(
        METRICS_PLOT
    )


    print(
        "Error trade-off plot:"
    )

    print(
        ERRORS_PLOT
    )


    print("=" * 92)


if __name__ == "__main__":
    main()
