#!/usr/bin/env python3
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

DB_FILE = Path("/hydroponic_project/database/hydroponic.db")
OUTPUT_DIR = Path("/hydroponic_project/evaluation/multimodal_analysis")

def normalise_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "healthy":
        return "healthy"
    if text == "unhealthy" or "stress" in text:
        return "unhealthy"
    return None

def pct(a, b):
    return 0.0 if b == 0 else (a / b) * 100.0

def load_data():
    if not DB_FILE.exists():
        raise FileNotFoundError(f"Database not found: {DB_FILE}")

    conn = sqlite3.connect(str(DB_FILE))
    try:
        df = pd.read_sql_query(
            '''
            SELECT
                ce.capture_id,
                ce.timestamp,
                ce.ph,
                ce.ec_ms_cm,
                ce.tds_ppm,
                ce.water_temp_c,
                ce.air_temp_c,
                ce.humidity_pct,
                ce.water_level,
                ce.camera_lux,
                ps.plant_id,
                ps.plant_active,
                ps.image_prediction,
                ps.image_confidence,
                ps.sensor_prediction,
                ps.sensor_confidence,
                ps.fusion_prediction,
                ps.fusion_confidence,
                ps.final_status
            FROM plant_samples ps
            JOIN capture_events ce
              ON ce.capture_id = ps.capture_id
            WHERE ps.plant_active = 1
            ORDER BY ce.timestamp, ps.plant_id
            ''',
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("No active plant observations found.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    df["image_label"] = df["image_prediction"].apply(normalise_label)
    df["sensor_label"] = df["sensor_prediction"].apply(normalise_label)
    df["fusion_label"] = df["fusion_prediction"].apply(normalise_label)

    eligible = df[
        df["image_label"].notna()
        & df["sensor_label"].notna()
        & df["fusion_label"].notna()
    ].copy()

    if eligible.empty:
        raise RuntimeError("No rows contain valid Image, Sensor and Fusion predictions.")

    eligible["image_sensor_agree"] = (
        eligible["image_label"] == eligible["sensor_label"]
    )
    eligible["image_fusion_agree"] = (
        eligible["image_label"] == eligible["fusion_label"]
    )
    eligible["sensor_fusion_agree"] = (
        eligible["sensor_label"] == eligible["fusion_label"]
    )

    eligible["fusion_resolution"] = "not_applicable"
    disagree = ~eligible["image_sensor_agree"]

    eligible.loc[
        disagree & (eligible["fusion_label"] == eligible["image_label"]),
        "fusion_resolution",
    ] = "follows_image"

    eligible.loc[
        disagree & (eligible["fusion_label"] == eligible["sensor_label"]),
        "fusion_resolution",
    ] = "follows_sensor"

    disagreements = eligible[disagree].copy()

    total = len(eligible)
    unique_captures = eligible["capture_id"].nunique()
    is_agree = int(eligible["image_sensor_agree"].sum())
    is_disagree = total - is_agree
    if_agree = int(eligible["image_fusion_agree"].sum())
    sf_agree = int(eligible["sensor_fusion_agree"].sum())

    follows_image = int(
        (disagreements["fusion_resolution"] == "follows_image").sum()
    )
    follows_sensor = int(
        (disagreements["fusion_resolution"] == "follows_sensor").sum()
    )

    summary = pd.DataFrame(
        [
            ["Eligible plant observations", total, np.nan],
            ["Unique captures", unique_captures, np.nan],
            ["Image-Sensor agreements", is_agree, pct(is_agree, total)],
            ["Image-Sensor disagreements", is_disagree, pct(is_disagree, total)],
            ["Image-Fusion agreements", if_agree, pct(if_agree, total)],
            ["Sensor-Fusion agreements", sf_agree, pct(sf_agree, total)],
            [
                "Fusion follows Image during disagreement",
                follows_image,
                pct(follows_image, is_disagree),
            ],
            [
                "Fusion follows Sensor during disagreement",
                follows_sensor,
                pct(follows_sensor, is_disagree),
            ],
            [
                "Mean Image confidence",
                pd.to_numeric(
                    eligible["image_confidence"], errors="coerce"
                ).mean(),
                np.nan,
            ],
            [
                "Mean Sensor confidence",
                pd.to_numeric(
                    eligible["sensor_confidence"], errors="coerce"
                ).mean(),
                np.nan,
            ],
            [
                "Mean Fusion confidence",
                pd.to_numeric(
                    eligible["fusion_confidence"], errors="coerce"
                ).mean(),
                np.nan,
            ],
        ],
        columns=["metric", "value", "percentage"],
    )

    capture_summary = (
        eligible.groupby(["capture_id", "timestamp"], as_index=False)
        .agg(
            plant_observations=("plant_id", "count"),
            image_sensor_agreements=("image_sensor_agree", "sum"),
            image_fusion_agreements=("image_fusion_agree", "sum"),
            sensor_fusion_agreements=("sensor_fusion_agree", "sum"),
            ph=("ph", "first"),
            tds_ppm=("tds_ppm", "first"),
            air_temp_c=("air_temp_c", "first"),
            humidity_pct=("humidity_pct", "first"),
            water_temp_c=("water_temp_c", "first"),
            camera_lux=("camera_lux", "first"),
        )
    )
    capture_summary["image_sensor_disagreements"] = (
        capture_summary["plant_observations"]
        - capture_summary["image_sensor_agreements"]
    )

    eligible.to_csv(
        OUTPUT_DIR / "multimodal_observations.csv",
        index=False,
    )
    disagreements.to_csv(
        OUTPUT_DIR / "multimodal_disagreements.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "multimodal_summary.csv",
        index=False,
    )
    capture_summary.to_csv(
        OUTPUT_DIR / "multimodal_capture_summary.csv",
        index=False,
    )

    print()
    print("=" * 60)
    print(" MULTIMODAL AGREEMENT ANALYSIS")
    print("=" * 60)
    print(f"Eligible plant observations : {total}")
    print(f"Unique captures             : {unique_captures}")
    print()
    print(
        f"Image vs Sensor agree       : {is_agree} "
        f"({pct(is_agree, total):.2f}%)"
    )
    print(
        f"Image vs Sensor disagree    : {is_disagree} "
        f"({pct(is_disagree, total):.2f}%)"
    )
    print(
        f"Image vs Fusion agree       : {if_agree} "
        f"({pct(if_agree, total):.2f}%)"
    )
    print(
        f"Sensor vs Fusion agree      : {sf_agree} "
        f"({pct(sf_agree, total):.2f}%)"
    )
    print()
    print("During Image-Sensor disagreement:")
    print(
        f"Fusion follows Image        : {follows_image} "
        f"({pct(follows_image, is_disagree):.2f}%)"
    )
    print(
        f"Fusion follows Sensor       : {follows_sensor} "
        f"({pct(follows_sensor, is_disagree):.2f}%)"
    )
    print()
    print("IMPORTANT: These are agreement statistics, NOT accuracy statistics.")
    print("Ground truth is required later to determine which model is correct.")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
