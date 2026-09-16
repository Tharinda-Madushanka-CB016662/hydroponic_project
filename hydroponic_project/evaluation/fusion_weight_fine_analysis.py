#!/usr/bin/env python3

# ============================================================
# FINE-GRAINED LATE-FUSION WEIGHT ANALYSIS
# ============================================================
#
# Purpose
# -------
# Refine the broad 0.10-step sensitivity experiment around
# the transition region where fusion changes from mostly
# following the Sensor model to mostly following the Image
# model.
#
# Two scans are performed:
#
#   Scan A: 0.40 -> 0.50 image weight, step 0.01
#   Scan B: 0.470 -> 0.500 image weight, step 0.001
#
# Sensor weight is always:
#
#   1.0 - image_weight
#
# IMPORTANT
# ---------
# This identifies the most behaviourally balanced weighting.
# It does NOT identify the most accurate weighting because
# no independent ground truth is used.
#
# ============================================================

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/fusion_sensitivity"
)

COARSE_FILE = (
    OUTPUT_DIR / "fusion_fine_040_050_step_001.csv"
)

FINE_FILE = (
    OUTPUT_DIR / "fusion_fine_0470_0500_step_0001.csv"
)

DEFAULT_THRESHOLD = 0.50


def normalise_label(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text == "healthy":
        return "healthy"

    if text == "unhealthy" or "stress" in text:
        return "unhealthy"

    return None


def get_threshold(conn):
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
            return DEFAULT_THRESHOLD

        threshold = float(row[0])

        if 0.0 < threshold < 1.0:
            return threshold

    except Exception:
        pass

    return DEFAULT_THRESHOLD


def predicted_confidence_to_risk(label, confidence):
    confidence = float(confidence)

    if label == "unhealthy":
        return confidence

    if label == "healthy":
        return 1.0 - confidence

    return np.nan


def load_data():
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}"
        )

    conn = sqlite3.connect(str(DB_FILE))

    try:
        threshold = get_threshold(conn)

        df = pd.read_sql_query(
            """
            SELECT
                ce.capture_id,
                ce.timestamp,
                ps.plant_id,
                ps.image_prediction,
                ps.image_confidence,
                ps.sensor_prediction,
                ps.sensor_confidence

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

    df["image_label"] = (
        df["image_prediction"]
        .apply(normalise_label)
    )

    df["sensor_label"] = (
        df["sensor_prediction"]
        .apply(normalise_label)
    )

    df["image_confidence"] = pd.to_numeric(
        df["image_confidence"],
        errors="coerce",
    )

    df["sensor_confidence"] = pd.to_numeric(
        df["sensor_confidence"],
        errors="coerce",
    )

    df = df[
        df["image_label"].notna()
        & df["sensor_label"].notna()
        & df["image_confidence"].between(0, 1)
        & df["sensor_confidence"].between(0, 1)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No eligible Image + Sensor observations found."
        )

    df["image_risk"] = [
        predicted_confidence_to_risk(label, conf)
        for label, conf in zip(
            df["image_label"],
            df["image_confidence"],
        )
    ]

    df["sensor_risk"] = [
        predicted_confidence_to_risk(label, conf)
        for label, conf in zip(
            df["sensor_label"],
            df["sensor_confidence"],
        )
    ]

    # We only need disagreement observations to find the
    # behavioural balance point.
    disagreements = df[
        df["image_label"]
        != df["sensor_label"]
    ].copy()

    if disagreements.empty:
        raise RuntimeError(
            "Image and Sensor never disagree in eligible data."
        )

    return (
        disagreements,
        threshold,
    )


def pct(a, b):
    if b == 0:
        return 0.0

    return float(a) / float(b) * 100.0


def analyse_weights(
    disagreements,
    threshold,
    weights,
):
    rows = []

    total = len(disagreements)

    for image_weight in weights:
        image_weight = float(
            image_weight
        )

        sensor_weight = (
            1.0 - image_weight
        )

        fusion_risk = (
            image_weight
            * disagreements["image_risk"]
            +
            sensor_weight
            * disagreements["sensor_risk"]
        )

        fusion_label = np.where(
            fusion_risk >= threshold,
            "unhealthy",
            "healthy",
        )

        follows_image = int(
            (
                fusion_label
                == disagreements[
                    "image_label"
                ].to_numpy()
            ).sum()
        )

        follows_sensor = int(
            (
                fusion_label
                == disagreements[
                    "sensor_label"
                ].to_numpy()
            ).sum()
        )

        image_pct = pct(
            follows_image,
            total,
        )

        sensor_pct = pct(
            follows_sensor,
            total,
        )

        balance_gap = abs(
            image_pct
            - sensor_pct
        )

        rows.append(
            {
                "image_weight":
                    image_weight,

                "sensor_weight":
                    sensor_weight,

                "disagreement_observations":
                    total,

                "fusion_follows_image_count":
                    follows_image,

                "fusion_follows_image_pct":
                    image_pct,

                "fusion_follows_sensor_count":
                    follows_sensor,

                "fusion_follows_sensor_pct":
                    sensor_pct,

                "balance_gap_percentage_points":
                    balance_gap,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_table(title, df):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    display = df[
        [
            "image_weight",
            "sensor_weight",
            "fusion_follows_image_pct",
            "fusion_follows_sensor_pct",
            "balance_gap_percentage_points",
        ]
    ].copy()

    display.columns = [
        "Img W",
        "Sensor W",
        "Follows Image %",
        "Follows Sensor %",
        "Balance Gap pp",
    ]

    print(
        display.to_string(
            index=False,
            formatters={
                "Img W":
                    lambda x: f"{x:.3f}",

                "Sensor W":
                    lambda x: f"{x:.3f}",

                "Follows Image %":
                    lambda x: f"{x:.2f}",

                "Follows Sensor %":
                    lambda x: f"{x:.2f}",

                "Balance Gap pp":
                    lambda x: f"{x:.2f}",
            },
        )
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        disagreements,
        threshold,
    ) = load_data()

    print()
    print("=" * 88)
    print(" FINE-GRAINED LATE-FUSION WEIGHT ANALYSIS")
    print("=" * 88)

    print(
        f"Decision threshold          : {threshold:.2f}"
    )

    print(
        f"Image-Sensor disagreements : {len(disagreements)}"
    )

    # --------------------------------------------------------
    # Scan A: 0.40 -> 0.50 by 0.01
    # --------------------------------------------------------

    weights_a = np.round(
        np.arange(
            0.40,
            0.501,
            0.01,
        ),
        3,
    )

    coarse = analyse_weights(
        disagreements,
        threshold,
        weights_a,
    )

    coarse.to_csv(
        COARSE_FILE,
        index=False,
    )

    print_table(
        "SCAN A: IMAGE WEIGHT 0.40 -> 0.50 (STEP 0.01)",
        coarse,
    )

    # --------------------------------------------------------
    # Scan B: 0.470 -> 0.500 by 0.001
    # --------------------------------------------------------

    weights_b = np.round(
        np.arange(
            0.470,
            0.5001,
            0.001,
        ),
        3,
    )

    fine = analyse_weights(
        disagreements,
        threshold,
        weights_b,
    )

    fine.to_csv(
        FINE_FILE,
        index=False,
    )

    best_index = (
        fine[
            "balance_gap_percentage_points"
        ].idxmin()
    )

    best = fine.loc[
        best_index
    ]

    print_table(
        "SCAN B: IMAGE WEIGHT 0.470 -> 0.500 (STEP 0.001)",
        fine,
    )

    print()
    print("=" * 88)
    print(" CLOSEST BEHAVIOURAL BALANCE")
    print("=" * 88)

    print(
        f"Image weight          : "
        f"{best['image_weight']:.3f}"
    )

    print(
        f"Sensor weight         : "
        f"{best['sensor_weight']:.3f}"
    )

    print(
        f"Fusion follows Image  : "
        f"{best['fusion_follows_image_pct']:.2f}%"
    )

    print(
        f"Fusion follows Sensor : "
        f"{best['fusion_follows_sensor_pct']:.2f}%"
    )

    print(
        f"Balance gap           : "
        f"{best['balance_gap_percentage_points']:.2f} "
        f"percentage points"
    )

    print()
    print(
        "IMPORTANT: This is the closest behavioural balance,"
    )

    print(
        "not proof that this weighting is the most accurate."
    )

    print(
        "Ground-truth evaluation is required before selecting"
    )

    print(
        "the final fusion weighting."
    )

    print()
    print(
        f"Scan A CSV: {COARSE_FILE}"
    )

    print(
        f"Scan B CSV: {FINE_FILE}"
    )

    print("=" * 88)
    print()


if __name__ == "__main__":
    main()
