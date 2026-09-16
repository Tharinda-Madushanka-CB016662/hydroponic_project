#!/usr/bin/env python3

# ============================================================
# HYDROPONIC LATE-FUSION WEIGHT SENSITIVITY ANALYSIS
# ============================================================
#
# Purpose
# -------
# Test how different Image/Sensor fusion weights change the
# behaviour of the late-fusion model WITHOUT changing the live
# production settings.
#
# This is a behavioural/sensitivity analysis, NOT an accuracy
# evaluation, because independent ground-truth labels are not
# used here.
#
# Tested image weights:
#   0.00, 0.10, 0.20, ... 1.00
#
# Sensor weight is:
#   1.00 - image_weight
#
# For each weight pair the script reports:
#   - total eligible plant observations
#   - Image-Fusion agreement
#   - Sensor-Fusion agreement
#   - Image-Sensor disagreement count
#   - during disagreement, Fusion follows Image %
#   - during disagreement, Fusion follows Sensor %
#   - Healthy / Unhealthy fusion prediction counts
#   - mean fusion confidence
#
# It also validates the current 0.60 / 0.40 calculation against
# the stored fusion predictions when possible.
#
# Outputs
# -------
# /hydroponic_project/evaluation/fusion_sensitivity/
#
#   fusion_weight_sensitivity.csv
#   fusion_weight_sensitivity_disagreements.csv
#
# ============================================================

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/fusion_sensitivity"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "fusion_weight_sensitivity.csv"
)

DISAGREEMENT_FILE = (
    OUTPUT_DIR / "fusion_weight_sensitivity_disagreements.csv"
)


# ============================================================
# DEFAULT DECISION THRESHOLD
# ============================================================
#
# If system_settings contains decision_threshold, that value
# will be used instead.
# ============================================================

DEFAULT_DECISION_THRESHOLD = 0.50


# ============================================================
# NORMALISE MODEL LABELS
# ============================================================

def normalise_label(value):
    """
    Convert model text into a common binary representation.

        healthy                 -> healthy
        unhealthy               -> unhealthy
        stress / stress warning -> unhealthy

    Returns None for skipped/unavailable results.
    """

    if value is None:
        return None

    text = str(value).strip().lower()

    if text == "healthy":
        return "healthy"

    if (
        text == "unhealthy"
        or "stress" in text
    ):
        return "unhealthy"

    return None


# ============================================================
# GET DECISION THRESHOLD
# ============================================================

def get_decision_threshold(conn):
    """
    Read decision_threshold from system_settings if the table
    and setting exist. Otherwise use 0.50.
    """

    try:
        row = conn.execute(
            """
            SELECT setting_value
            FROM system_settings
            WHERE setting_key = 'decision_threshold'
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            return DEFAULT_DECISION_THRESHOLD

        value = float(row[0])

        if not 0.0 < value < 1.0:
            return DEFAULT_DECISION_THRESHOLD

        return value

    except Exception:
        return DEFAULT_DECISION_THRESHOLD


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load active plant rows that contain Image and Sensor model
    predictions and their predicted-class confidences.
    """

    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}"
        )

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    try:
        threshold = get_decision_threshold(
            conn
        )

        df = pd.read_sql_query(
            """
            SELECT
                ce.capture_id,
                ce.timestamp,
                ce.camera_lux,

                ps.plant_id,
                ps.plant_active,

                ps.image_prediction,
                ps.image_confidence,

                ps.sensor_prediction,
                ps.sensor_confidence,

                ps.fusion_prediction,
                ps.fusion_confidence

            FROM plant_samples ps

            JOIN capture_events ce
              ON ce.capture_id = ps.capture_id

            WHERE ps.plant_active = 1

            ORDER BY
                ce.timestamp,
                ps.plant_id
            """,
            conn,
        )

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "No active plant observations found."
        )

    df["image_label"] = (
        df["image_prediction"]
        .apply(normalise_label)
    )

    df["sensor_label"] = (
        df["sensor_prediction"]
        .apply(normalise_label)
    )

    df["stored_fusion_label"] = (
        df["fusion_prediction"]
        .apply(normalise_label)
    )

    # Keep rows where Image and Sensor are both available.
    # This automatically excludes low-light/nighttime rows.
    eligible = df[
        df["image_label"].notna()
        & df["sensor_label"].notna()
        & df["image_confidence"].notna()
        & df["sensor_confidence"].notna()
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible Image + Sensor observations found."
        )

    eligible["image_confidence"] = pd.to_numeric(
        eligible["image_confidence"],
        errors="coerce",
    )

    eligible["sensor_confidence"] = pd.to_numeric(
        eligible["sensor_confidence"],
        errors="coerce",
    )

    eligible = eligible[
        eligible["image_confidence"].between(0, 1)
        & eligible["sensor_confidence"].between(0, 1)
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No valid confidence values were found."
        )

    return eligible, threshold


# ============================================================
# CONVERT PREDICTED-CLASS CONFIDENCE TO UNHEALTHY RISK
# ============================================================

def predicted_confidence_to_risk(
    label,
    confidence,
):
    """
    Current database confidence fields store confidence in the
    predicted class.

    Therefore:

        predicted unhealthy:
            unhealthy risk = confidence

        predicted healthy:
            unhealthy risk = 1 - confidence
    """

    confidence = float(
        confidence
    )

    if label == "unhealthy":
        return confidence

    if label == "healthy":
        return 1.0 - confidence

    return np.nan


# ============================================================
# RECONSTRUCT INPUT RISKS
# ============================================================

def add_risk_probabilities(df):
    result = df.copy()

    result["image_risk"] = [
        predicted_confidence_to_risk(
            label,
            conf,
        )
        for label, conf in zip(
            result["image_label"],
            result["image_confidence"],
        )
    ]

    result["sensor_risk"] = [
        predicted_confidence_to_risk(
            label,
            conf,
        )
        for label, conf in zip(
            result["sensor_label"],
            result["sensor_confidence"],
        )
    ]

    return result


# ============================================================
# FUSION CALCULATION
# ============================================================

def calculate_fusion(
    df,
    image_weight,
    threshold,
):
    """
    Calculate late-fusion unhealthy risk:

        fusion_risk =
            image_weight * image_risk
          + sensor_weight * sensor_risk

    Decision:
        risk >= threshold -> unhealthy
        risk < threshold  -> healthy
    """

    sensor_weight = (
        1.0
        - image_weight
    )

    fusion_risk = (
        image_weight
        * df["image_risk"]
        +
        sensor_weight
        * df["sensor_risk"]
    )

    fusion_label = np.where(
        fusion_risk >= threshold,
        "unhealthy",
        "healthy",
    )

    # Confidence in whichever class was predicted.
    fusion_confidence = np.where(
        fusion_label == "unhealthy",
        fusion_risk,
        1.0 - fusion_risk,
    )

    return (
        fusion_risk,
        fusion_label,
        fusion_confidence,
    )


# ============================================================
# SAFE PERCENTAGE
# ============================================================

def pct(a, b):
    if b == 0:
        return 0.0

    return (
        float(a)
        / float(b)
        * 100.0
    )


# ============================================================
# RUN SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity(
    df,
    threshold,
):
    """
    Evaluate image weights from 0.00 to 1.00 in 0.10 steps.
    """

    rows = []
    disagreement_rows = []

    image_sensor_disagreement = (
        df["image_label"]
        != df["sensor_label"]
    )

    disagreement_count = int(
        image_sensor_disagreement.sum()
    )

    total = len(
        df
    )

    unique_captures = (
        df["capture_id"]
        .nunique()
    )

    # 0.00, 0.10, ..., 1.00
    weights = np.round(
        np.arange(
            0.0,
            1.01,
            0.10,
        ),
        2,
    )

    for image_weight in weights:
        sensor_weight = round(
            1.0
            - float(image_weight),
            2,
        )

        (
            fusion_risk,
            fusion_label,
            fusion_confidence,
        ) = calculate_fusion(
            df,
            float(image_weight),
            threshold,
        )

        temp = df.copy()

        temp["tested_image_weight"] = (
            float(image_weight)
        )

        temp["tested_sensor_weight"] = (
            sensor_weight
        )

        temp["tested_fusion_risk"] = (
            fusion_risk
        )

        temp["tested_fusion_label"] = (
            fusion_label
        )

        temp["tested_fusion_confidence"] = (
            fusion_confidence
        )

        image_fusion_agree = (
            temp["image_label"]
            == temp["tested_fusion_label"]
        )

        sensor_fusion_agree = (
            temp["sensor_label"]
            == temp["tested_fusion_label"]
        )

        disagreement_temp = temp[
            image_sensor_disagreement
        ].copy()

        follows_image = (
            disagreement_temp[
                "tested_fusion_label"
            ]
            == disagreement_temp[
                "image_label"
            ]
        )

        follows_sensor = (
            disagreement_temp[
                "tested_fusion_label"
            ]
            == disagreement_temp[
                "sensor_label"
            ]
        )

        healthy_count = int(
            (
                temp[
                    "tested_fusion_label"
                ] == "healthy"
            ).sum()
        )

        unhealthy_count = (
            total
            - healthy_count
        )

        rows.append(
            {
                "image_weight":
                    float(image_weight),

                "sensor_weight":
                    sensor_weight,

                "decision_threshold":
                    threshold,

                "plant_observations":
                    total,

                "unique_captures":
                    unique_captures,

                "image_sensor_disagreements":
                    disagreement_count,

                "image_fusion_agreement_pct":
                    pct(
                        int(
                            image_fusion_agree.sum()
                        ),
                        total,
                    ),

                "sensor_fusion_agreement_pct":
                    pct(
                        int(
                            sensor_fusion_agree.sum()
                        ),
                        total,
                    ),

                "fusion_follows_image_count":
                    int(
                        follows_image.sum()
                    ),

                "fusion_follows_image_pct":
                    pct(
                        int(
                            follows_image.sum()
                        ),
                        disagreement_count,
                    ),

                "fusion_follows_sensor_count":
                    int(
                        follows_sensor.sum()
                    ),

                "fusion_follows_sensor_pct":
                    pct(
                        int(
                            follows_sensor.sum()
                        ),
                        disagreement_count,
                    ),

                "fusion_healthy_count":
                    healthy_count,

                "fusion_unhealthy_count":
                    unhealthy_count,

                "mean_fusion_confidence":
                    float(
                        np.mean(
                            fusion_confidence
                        )
                    ),

                "mean_fusion_risk":
                    float(
                        np.mean(
                            fusion_risk
                        )
                    ),
            }
        )

        # Save only disagreement rows for detailed inspection.
        disagreement_temp[
            "fusion_follows"
        ] = np.where(
            follows_image,
            "image",
            np.where(
                follows_sensor,
                "sensor",
                "other",
            ),
        )

        disagreement_rows.append(
            disagreement_temp[
                [
                    "capture_id",
                    "timestamp",
                    "plant_id",
                    "camera_lux",
                    "image_label",
                    "image_confidence",
                    "image_risk",
                    "sensor_label",
                    "sensor_confidence",
                    "sensor_risk",
                    "tested_image_weight",
                    "tested_sensor_weight",
                    "tested_fusion_risk",
                    "tested_fusion_label",
                    "tested_fusion_confidence",
                    "fusion_follows",
                ]
            ]
        )

    summary = pd.DataFrame(
        rows
    )

    details = pd.concat(
        disagreement_rows,
        ignore_index=True,
    )

    return (
        summary,
        details,
    )


# ============================================================
# VALIDATE CURRENT 60 / 40 CONFIGURATION
# ============================================================

def validate_current_60_40(
    df,
    threshold,
):
    """
    Compare recomputed 60/40 labels with stored fusion labels.

    This helps verify that confidence-to-risk reconstruction
    matches the production fusion implementation.
    """

    comparable = df[
        df["stored_fusion_label"].notna()
    ].copy()

    if comparable.empty:
        return None

    (
        _,
        calculated_label,
        _,
    ) = calculate_fusion(
        comparable,
        0.60,
        threshold,
    )

    matches = (
        comparable[
            "stored_fusion_label"
        ].to_numpy()
        == calculated_label
    )

    match_count = int(
        matches.sum()
    )

    total = len(
        comparable
    )

    return {
        "total":
            total,

        "matches":
            match_count,

        "mismatches":
            total - match_count,

        "match_pct":
            pct(
                match_count,
                total,
            ),
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_results(
    summary,
    threshold,
    validation,
):
    print()
    print(
        "=" * 72
    )

    print(
        " LATE-FUSION WEIGHT SENSITIVITY ANALYSIS"
    )

    print(
        "=" * 72
    )

    print(
        f"Decision threshold : {threshold:.2f}"
    )

    print()

    display_columns = [
        "image_weight",
        "sensor_weight",
        "image_fusion_agreement_pct",
        "sensor_fusion_agreement_pct",
        "fusion_follows_image_pct",
        "fusion_follows_sensor_pct",
        "mean_fusion_confidence",
    ]

    display = summary[
        display_columns
    ].copy()

    display.columns = [
        "Img W",
        "Sensor W",
        "Img-Fusion %",
        "Sensor-Fusion %",
        "Follows Img %",
        "Follows Sensor %",
        "Mean Fusion Conf",
    ]

    print(
        display.to_string(
            index=False,
            formatters={
                "Img W":
                    lambda x: f"{x:.2f}",

                "Sensor W":
                    lambda x: f"{x:.2f}",

                "Img-Fusion %":
                    lambda x: f"{x:.2f}",

                "Sensor-Fusion %":
                    lambda x: f"{x:.2f}",

                "Follows Img %":
                    lambda x: f"{x:.2f}",

                "Follows Sensor %":
                    lambda x: f"{x:.2f}",

                "Mean Fusion Conf":
                    lambda x: f"{x:.4f}",
            },
        )
    )

    print()

    if validation is not None:
        print(
            "60/40 production-formula validation"
        )

        print(
            f"  Comparable rows : "
            f"{validation['total']}"
        )

        print(
            f"  Matches         : "
            f"{validation['matches']}"
        )

        print(
            f"  Mismatches      : "
            f"{validation['mismatches']}"
        )

        print(
            f"  Match rate      : "
            f"{validation['match_pct']:.2f}%"
        )

        print()

    print(
        "IMPORTANT:"
    )

    print(
        "This analysis shows how fusion behaviour changes "
        "with weighting."
    )

    print(
        "It does NOT determine which weighting is most accurate."
    )

    print(
        "Independent ground truth is required for that later step."
    )

    print()

    print(
        f"Summary CSV      : {SUMMARY_FILE}"
    )

    print(
        f"Disagreement CSV : {DISAGREEMENT_FILE}"
    )

    print(
        "=" * 72
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        df,
        threshold,
    ) = load_data()

    df = add_risk_probabilities(
        df
    )

    validation = (
        validate_current_60_40(
            df,
            threshold,
        )
    )

    (
        summary,
        details,
    ) = run_sensitivity(
        df,
        threshold,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    details.to_csv(
        DISAGREEMENT_FILE,
        index=False,
    )

    print_results(
        summary,
        threshold,
        validation,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
