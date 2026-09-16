#!/usr/bin/env python3

# ============================================================
# NEW IMAGE vs SENSOR vs LATE-FUSION GROUND-TRUTH EVALUATION
# ============================================================
#
# Purpose
# -------
# Compare the NEW MobileNetV2 image model, the existing sensor
# Random Forest, and late fusion against the SAME manually
# labelled Raspberry Pi observations.
#
# This is the core multimodal evaluation step.
#
# IMPORTANT:
#   - READ-ONLY with respect to SQLite.
#   - Does NOT modify production predictions.
#   - Uses NEW MobileNetV2 unhealthy probabilities saved by
#     evaluate_new_image_models_live_ground_truth.py.
#   - Re-runs the sensor Random Forest directly from the
#     capture-level sensor values so we obtain the exact
#     unhealthy probability rather than reconstructing it from
#     rounded confidence values.
#
# Class semantics
# ---------------
# Image model:
#   healthy   = 0
#   unhealthy = 1
#
# Sensor model:
#   unhealthy/stress = class 0
#   healthy          = class 1
#
# Therefore:
#   image_risk  = P(image class 1 = unhealthy)
#   sensor_risk = P(sensor class 0 = unhealthy/stress)
#
# Fusion formula
# --------------
#   fusion_risk =
#       image_weight * image_risk
#       +
#       sensor_weight * sensor_risk
#
# where:
#   sensor_weight = 1 - image_weight
#
# Default fusion decision threshold:
#   0.50
#
# The image-only evaluation is shown at:
#   0.50 = original/default threshold
#   0.42 = exploratory candidate from threshold sensitivity
#
# Outputs
# -------
# /hydroponic_project/evaluation/final_multimodal_ground_truth/
#
#   multimodal_observation_predictions.csv
#   image_sensor_baseline_metrics.csv
#   fusion_weight_sensitivity.csv
#   fusion_best_candidates.csv
#   fusion_weight_metrics.png
# ============================================================

import sqlite3
from pathlib import Path

import joblib
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

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

LIVE_PREDICTIONS_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)

SENSOR_MODEL_PACKAGE = Path(
    "/hydroponic_project/models/sensor/"
    "sensor_model_package.joblib"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/"
    "final_multimodal_ground_truth"
)

OBSERVATION_CSV = (
    OUTPUT_DIR
    / "multimodal_observation_predictions.csv"
)

BASELINE_METRICS_CSV = (
    OUTPUT_DIR
    / "image_sensor_baseline_metrics.csv"
)

FUSION_CSV = (
    OUTPUT_DIR
    / "fusion_weight_sensitivity.csv"
)

BEST_CSV = (
    OUTPUT_DIR
    / "fusion_best_candidates.csv"
)

FUSION_PLOT = (
    OUTPUT_DIR
    / "fusion_weight_metrics.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_DEFAULT_THRESHOLD = 0.50

IMAGE_CANDIDATE_THRESHOLD = 0.42

SENSOR_THRESHOLD = 0.50

FUSION_THRESHOLD = 0.50

# Scan image weight from 0.00 to 1.00 in 0.01 increments.
WEIGHT_STEP = 0.01

IMAGE_PROBABILITY_COLUMN = (
    "New_MobileNetV2_unhealthy_probability"
)


# ============================================================
# SENSOR FEATURE ORDER
# ============================================================
#
# This must match the feature order used during sensor-model
# training.
# ============================================================

SENSOR_FEATURES = [
    "pH",
    "TDS",
    "water_level",
    "DHT_temp",
    "DHT_humidity",
    "water_temp",
]


# ============================================================
# HELPERS
# ============================================================

def encode_water_level(
    value,
):
    """
    Convert the capture water-level text to the numeric feature
    used by the sensor Random Forest.

    Existing project convention:
        OK / normal / high -> 1
        LOW               -> 0
    """

    if pd.isna(
        value
    ):
        return np.nan

    text = (
        str(
            value
        )
        .strip()
        .upper()
    )

    if text in {
        "OK",
        "NORMAL",
        "HIGH",
        "1",
        "TRUE",
    }:
        return 1.0

    if text in {
        "LOW",
        "0",
        "FALSE",
    }:
        return 0.0

    raise ValueError(
        f"Unknown water_level value: {value}"
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Positive class:
        unhealthy
    """

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


def prediction_from_risk(
    risk_values,
    threshold,
):
    return np.where(
        np.asarray(
            risk_values,
            dtype=float,
        )
        >= threshold,
        "unhealthy",
        "healthy",
    )


# ============================================================
# LOAD LIVE GROUND TRUTH + NEW IMAGE PROBABILITIES
# ============================================================

def load_live_predictions():

    if not LIVE_PREDICTIONS_CSV.exists():

        raise FileNotFoundError(
            f"Live prediction CSV not found:\n"
            f"{LIVE_PREDICTIONS_CSV}"
        )

    df = pd.read_csv(
        LIVE_PREDICTIONS_CSV
    )

    required = {
        "capture_id",
        "plant_id",
        "ground_truth",
        IMAGE_PROBABILITY_COLUMN,
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing required live-prediction columns: "
            f"{sorted(missing)}"
        )

    df[
        "ground_truth"
    ] = (
        df[
            "ground_truth"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = (
        df[
            df[
                "ground_truth"
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

    if df[
        IMAGE_PROBABILITY_COLUMN
    ].isna().any():

        raise RuntimeError(
            "Invalid New MobileNetV2 unhealthy "
            "probabilities were found."
        )

    return df


# ============================================================
# LOAD CAPTURE-LEVEL SENSOR VALUES
# ============================================================

def load_capture_sensors():

    if not DB_FILE.exists():

        raise FileNotFoundError(
            f"Database not found: "
            f"{DB_FILE}"
        )

    conn = sqlite3.connect(
        str(
            DB_FILE
        )
    )

    try:

        sensor_df = pd.read_sql_query(
            """
            SELECT
                capture_id,
                timestamp,
                ph,
                tds_ppm,
                water_level,
                air_temp_c,
                humidity_pct,
                water_temp_c

            FROM capture_events
            """,
            conn,
        )

    finally:

        conn.close()

    sensor_df[
        "water_level_numeric"
    ] = sensor_df[
        "water_level"
    ].apply(
        encode_water_level
    )

    return sensor_df


# ============================================================
# SENSOR MODEL
# ============================================================

def load_sensor_model():

    if not SENSOR_MODEL_PACKAGE.exists():

        raise FileNotFoundError(
            f"Sensor model package not found:\n"
            f"{SENSOR_MODEL_PACKAGE}"
        )

    package = joblib.load(
        SENSOR_MODEL_PACKAGE
    )

    # Support common packaging formats.
    if hasattr(
        package,
        "predict_proba",
    ):

        model = package

        package_features = None

    elif isinstance(
        package,
        dict,
    ):

        model = None

        for key in [
            "model",
            "classifier",
            "rf_model",
            "random_forest",
        ]:

            candidate = package.get(
                key
            )

            if (
                candidate is not None
                and hasattr(
                    candidate,
                    "predict_proba",
                )
            ):

                model = candidate
                break

        if model is None:

            raise RuntimeError(
                "Could not locate a predict_proba-capable "
                "model inside sensor_model_package.joblib."
            )

        package_features = (
            package.get(
                "features"
            )
            or package.get(
                "feature_names"
            )
            or package.get(
                "feature_columns"
            )
        )

    else:

        raise RuntimeError(
            "Unsupported sensor model package format."
        )

    print(
        "Sensor model classes :",
        model.classes_
    )

    print(
        "Expected features    :",
        SENSOR_FEATURES
    )

    if package_features is not None:

        print(
            "Package features     :",
            list(
                package_features
            )
        )

    # Sensor model convention from this project:
    #   class 0 = unhealthy / stress risk
    #   class 1 = healthy
    classes = list(
        model.classes_
    )

    if (
        0 not in classes
        or 1 not in classes
    ):

        raise RuntimeError(
            f"Unexpected sensor classes: "
            f"{classes}"
        )

    unhealthy_class_index = (
        classes.index(
            0
        )
    )

    return (
        model,
        unhealthy_class_index,
    )


def add_sensor_probabilities(
    df,
):

    model, unhealthy_index = (
        load_sensor_model()
    )

    feature_df = pd.DataFrame(
        {
            "pH":
                df[
                    "ph"
                ],

            "TDS":
                df[
                    "tds_ppm"
                ],

            "water_level":
                df[
                    "water_level_numeric"
                ],

            "DHT_temp":
                df[
                    "air_temp_c"
                ],

            "DHT_humidity":
                df[
                    "humidity_pct"
                ],

            "water_temp":
                df[
                    "water_temp_c"
                ],
        }
    )

    if feature_df.isna().any().any():

        bad_rows = feature_df[
            feature_df.isna().any(
                axis=1
            )
        ]

        raise RuntimeError(
            "Missing sensor values exist in manually labelled "
            f"observations. Bad rows: {len(bad_rows)}"
        )

    probabilities = (
        model.predict_proba(
            feature_df[
                SENSOR_FEATURES
            ]
        )
    )

    df[
        "sensor_unhealthy_probability"
    ] = probabilities[
        :,
        unhealthy_index
    ]

    return df


# ============================================================
# BASELINE EXPERIMENTS
# ============================================================

def evaluate_baselines(
    df,
):

    y_true = (
        df[
            "ground_truth"
        ]
        .to_numpy()
    )

    rows = []

    baseline_specs = [
        (
            "E1_Image_Only_Default_0.50",
            df[
                IMAGE_PROBABILITY_COLUMN
            ],
            IMAGE_DEFAULT_THRESHOLD,
        ),

        (
            "E1b_Image_Only_Candidate_0.42",
            df[
                IMAGE_PROBABILITY_COLUMN
            ],
            IMAGE_CANDIDATE_THRESHOLD,
        ),

        (
            "E2_Sensor_Only_0.50",
            df[
                "sensor_unhealthy_probability"
            ],
            SENSOR_THRESHOLD,
        ),
    ]

    for (
        experiment,
        risk,
        threshold,
    ) in baseline_specs:

        y_pred = (
            prediction_from_risk(
                risk,
                threshold,
            )
        )

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        row = {
            "experiment":
                experiment,

            "threshold":
                threshold,

            "samples":
                len(
                    df
                ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FUSION WEIGHT SENSITIVITY
# ============================================================

def evaluate_fusion_weights(
    df,
):

    y_true = (
        df[
            "ground_truth"
        ]
        .to_numpy()
    )

    image_risk = (
        df[
            IMAGE_PROBABILITY_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    sensor_risk = (
        df[
            "sensor_unhealthy_probability"
        ]
        .to_numpy(
            dtype=float
        )
    )

    rows = []

    weights = np.arange(
        0.0,
        1.0
        + (
            WEIGHT_STEP
            / 2.0
        ),
        WEIGHT_STEP,
    )

    for image_weight in (
        weights
    ):

        image_weight = round(
            float(
                image_weight
            ),
            2,
        )

        sensor_weight = round(
            1.0
            - image_weight,
            2,
        )

        fusion_risk = (
            image_weight
            * image_risk
            +
            sensor_weight
            * sensor_risk
        )

        y_pred = (
            prediction_from_risk(
                fusion_risk,
                FUSION_THRESHOLD,
            )
        )

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        row = {
            "image_weight":
                image_weight,

            "sensor_weight":
                sensor_weight,

            "fusion_threshold":
                FUSION_THRESHOLD,

            "samples":
                len(
                    df
                ),
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# BEST CANDIDATES
# ============================================================

def select_best(
    fusion_df,
    metric,
):

    temp = (
        fusion_df
        .copy()
    )

    temp[
        "distance_from_equal"
    ] = (
        temp[
            "image_weight"
        ]
        - 0.50
    ).abs()

    temp = (
        temp
        .sort_values(
            [
                metric,
                "balanced_accuracy",
                "unhealthy_f1",
                "unhealthy_recall",
                "healthy_as_unhealthy_fp",
                "distance_from_equal",
            ],
            ascending=[
                False,
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

    return temp.iloc[
        0
    ]


# ============================================================
# PLOT
# ============================================================

def create_fusion_plot(
    fusion_df,
):

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
        label="Unhealthy Recall",
    )

    plt.plot(
        fusion_df[
            "image_weight"
        ],
        fusion_df[
            "unhealthy_f1"
        ],
        label="Unhealthy F1",
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Equal 50/50 fusion",
    )

    plt.axvline(
        0.60,
        linestyle=":",
        label="Old production 60/40",
    )

    plt.xlabel(
        "Image weight"
    )

    plt.ylabel(
        "Metric value"
    )

    plt.title(
        "New MobileNetV2 + Sensor RF "
        "Late-Fusion Weight Sensitivity"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FUSION_PLOT,
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

    image_df = load_live_predictions()

    sensor_df = load_capture_sensors()

    # --------------------------------------------------------
    # MERGE MANUAL-LABEL / IMAGE RESULTS WITH CAPTURE SENSORS
    # --------------------------------------------------------
    #
    # live_image_model_predictions.csv already contains a
    # timestamp column, and capture_events also contains one.
    #
    # Without suffixes, pandas renames them to timestamp_x and
    # timestamp_y. The previous script then incorrectly looked
    # for a column called simply "timestamp", causing:
    #
    #     KeyError: 'timestamp'
    #
    # We therefore:
    #   1. explicitly suffix duplicate column names;
    #   2. use merge indicator to verify every labelled sample
    #      matched a capture_events row.
    # --------------------------------------------------------

    df = image_df.merge(
        sensor_df,
        on="capture_id",
        how="left",
        validate="many_to_one",
        suffixes=(
            "_live",
            "_sensor",
        ),
        indicator=True,
    )

    unmatched_mask = (
        df[
            "_merge"
        ]
        != "both"
    )

    if unmatched_mask.any():

        unmatched_capture_ids = (
            df.loc[
                unmatched_mask,
                "capture_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Some manually labelled observations could not be "
            "matched to capture_events. Capture IDs: "
            f"{unmatched_capture_ids}"
        )

    # Merge succeeded for every row.
    df = df.drop(
        columns=[
            "_merge"
        ]
    )

    # --------------------------------------------------------
    # VERIFY REQUIRED SENSOR VALUES
    # --------------------------------------------------------

    required_sensor_columns = [
        "ph",
        "tds_ppm",
        "water_level",
        "air_temp_c",
        "humidity_pct",
        "water_temp_c",
    ]

    missing_sensor_mask = (
        df[
            required_sensor_columns
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if missing_sensor_mask.any():

        bad_capture_ids = (
            df.loc[
                missing_sensor_mask,
                "capture_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Some matched captures contain missing sensor "
            "values required by the Random Forest. Capture IDs: "
            f"{bad_capture_ids}"
        )

    df = add_sensor_probabilities(
        df
    )

    # --------------------------------------------------------
    # SAVE OBSERVATION-LEVEL PROBABILITIES
    # --------------------------------------------------------

    df[
        "image_prediction_050"
    ] = prediction_from_risk(
        df[
            IMAGE_PROBABILITY_COLUMN
        ],
        IMAGE_DEFAULT_THRESHOLD,
    )

    df[
        "image_prediction_042"
    ] = prediction_from_risk(
        df[
            IMAGE_PROBABILITY_COLUMN
        ],
        IMAGE_CANDIDATE_THRESHOLD,
    )

    df[
        "sensor_prediction_050"
    ] = prediction_from_risk(
        df[
            "sensor_unhealthy_probability"
        ],
        SENSOR_THRESHOLD,
    )

    df.to_csv(
        OBSERVATION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    baseline_df = evaluate_baselines(
        df
    )

    baseline_df.to_csv(
        BASELINE_METRICS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fusion_df = evaluate_fusion_weights(
        df
    )

    fusion_df.to_csv(
        FUSION_CSV,
        index=False,
    )

    best_rows = []

    for (
        label,
        metric,
    ) in [
        (
            "best_macro_f1",
            "macro_f1",
        ),
        (
            "best_balanced_accuracy",
            "balanced_accuracy",
        ),
        (
            "best_unhealthy_f1",
            "unhealthy_f1",
        ),
        (
            "best_accuracy",
            "accuracy",
        ),
    ]:

        row = select_best(
            fusion_df,
            metric,
        )

        best_rows.append(
            {
                "selection":
                    label,

                **{
                    key:
                        row[
                            key
                        ]
                    for key
                    in fusion_df.columns
                },
            }
        )

    # Explicit reference weights.
    for (
        label,
        image_weight,
    ) in [
        (
            "equal_50_50",
            0.50,
        ),
        (
            "old_production_60_40",
            0.60,
        ),
    ]:

        row = (
            fusion_df[
                np.isclose(
                    fusion_df[
                        "image_weight"
                    ],
                    image_weight,
                )
            ]
            .iloc[0]
        )

        best_rows.append(
            {
                "selection":
                    label,

                **{
                    key:
                        row[
                            key
                        ]
                    for key
                    in fusion_df.columns
                },
            }
        )

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        BEST_CSV,
        index=False,
    )

    create_fusion_plot(
        fusion_df
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 108)
    print(
        " IMAGE vs SENSOR vs LATE-FUSION "
        "GROUND-TRUTH EVALUATION"
    )
    print("=" * 108)

    print(
        f"Binary observations : "
        f"{len(df)}"
    )

    print(
        f"Healthy             : "
        f"{int((df['ground_truth'] == 'healthy').sum())}"
    )

    print(
        f"Unhealthy           : "
        f"{int((df['ground_truth'] == 'unhealthy').sum())}"
    )

    print(
        f"Unique captures     : "
        f"{df['capture_id'].nunique()}"
    )

    print()

    print("=" * 108)
    print(
        " BASELINE EXPERIMENTS"
    )
    print("=" * 108)

    print(
        baseline_df[
            [
                "experiment",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "unhealthy_precision",
                "unhealthy_recall",
                "unhealthy_f1",
                "healthy_as_unhealthy_fp",
                "unhealthy_as_healthy_fn",
                "unhealthy_correct_tp",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print("=" * 108)
    print(
        " FUSION CANDIDATES"
    )
    print("=" * 108)

    print(
        best_df[
            [
                "selection",
                "image_weight",
                "sensor_weight",
                "fusion_threshold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "unhealthy_precision",
                "unhealthy_recall",
                "unhealthy_f1",
                "healthy_as_unhealthy_fp",
                "unhealthy_as_healthy_fn",
                "unhealthy_correct_tp",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The best fusion weighting on these same 149 labels "
        "is exploratory and must not be treated as independently "
        "validated production tuning."
    )

    print(
        "Use additional manually labelled future captures for "
        "final validation."
    )

    print()

    print(
        "Observation probabilities:"
    )
    print(
        OBSERVATION_CSV
    )

    print(
        "Baseline metrics:"
    )
    print(
        BASELINE_METRICS_CSV
    )

    print(
        "Fusion sensitivity:"
    )
    print(
        FUSION_CSV
    )

    print(
        "Best candidates:"
    )
    print(
        BEST_CSV
    )

    print(
        "Fusion plot:"
    )
    print(
        FUSION_PLOT
    )

    print("=" * 108)


if __name__ == "__main__":
    main()
