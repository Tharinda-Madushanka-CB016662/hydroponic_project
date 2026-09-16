#!/usr/bin/env python3
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DB_FILE = Path("/hydroponic_project/database/hydroponic.db")
OUTPUT_DIR = Path("/hydroponic_project/evaluation/ground_truth")
METRICS_FILE = OUTPUT_DIR / "preliminary_model_metrics.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "preliminary_predictions.csv"
CONFUSION_FILE = OUTPUT_DIR / "preliminary_confusion_matrices.csv"

EXPERIMENTS = {
    "E1_Image_Only": {"type": "image"},
    "E2_Sensor_Only": {"type": "sensor"},
    "E3_Fusion_0600_0400": {
        "type": "fusion",
        "image_weight": 0.600,
        "sensor_weight": 0.400,
    },
    "E4_Fusion_0488_0512": {
        "type": "fusion",
        "image_weight": 0.488,
        "sensor_weight": 0.512,
    },
    "E5_Fusion_0400_0600": {
        "type": "fusion",
        "image_weight": 0.400,
        "sensor_weight": 0.600,
    },
}

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
            '''
            SELECT setting_value
            FROM system_settings
            WHERE setting_key = 'decision_threshold'
            LIMIT 1
            '''
        ).fetchone()

        if row is not None:
            value = float(row[0])

            if 0.0 < value < 1.0:
                return value

    except Exception:
        pass

    return DEFAULT_THRESHOLD


def load_data():
    if not DB_FILE.exists():
        raise FileNotFoundError(f"Database not found: {DB_FILE}")

    conn = sqlite3.connect(str(DB_FILE))

    try:
        threshold = get_threshold(conn)

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

                ml.ground_truth_label,
                ml.validation_method,
                ml.reviewer_role,
                ml.notes,

                ps.image_prediction,
                ps.image_confidence,

                ps.sensor_prediction,
                ps.sensor_confidence,

                ps.fusion_prediction,
                ps.fusion_confidence

            FROM manual_labels ml

            INNER JOIN plant_samples ps
                ON ps.capture_id = ml.capture_id
               AND ps.plant_id = ml.plant_id

            INNER JOIN capture_events ce
                ON ce.capture_id = ml.capture_id

            WHERE
                ps.plant_active = 1
                AND LOWER(TRIM(ml.ground_truth_label))
                    IN ('healthy', 'unhealthy')

            ORDER BY
                ce.timestamp,
                ps.plant_id
            ''',
            conn,
        )

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "No Healthy/Unhealthy ground-truth labels found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["ground_truth"] = (
        df["ground_truth_label"]
        .astype(str)
        .str.strip()
        .str.lower()
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
            "No labelled observations have valid Image "
            "and Sensor predictions/confidences."
        )

    return df, threshold


def predicted_confidence_to_risk(label, confidence):
    confidence = float(confidence)

    if label == "unhealthy":
        return confidence

    if label == "healthy":
        return 1.0 - confidence

    return np.nan


def add_risks(df):
    result = df.copy()

    result["image_risk"] = [
        predicted_confidence_to_risk(label, conf)
        for label, conf in zip(
            result["image_label"],
            result["image_confidence"],
        )
    ]

    result["sensor_risk"] = [
        predicted_confidence_to_risk(label, conf)
        for label, conf in zip(
            result["sensor_label"],
            result["sensor_confidence"],
        )
    ]

    return result


def fusion_predict(df, image_weight, sensor_weight, threshold):
    risk = (
        image_weight * df["image_risk"]
        + sensor_weight * df["sensor_risk"]
    )

    label = np.where(
        risk >= threshold,
        "unhealthy",
        "healthy",
    )

    confidence = np.where(
        label == "unhealthy",
        risk,
        1.0 - risk,
    )

    return label, risk, confidence


def build_predictions(df, threshold):
    result = df.copy()

    result["E1_Image_Only"] = result["image_label"]
    result["E2_Sensor_Only"] = result["sensor_label"]

    for experiment_name in [
        "E3_Fusion_0600_0400",
        "E4_Fusion_0488_0512",
        "E5_Fusion_0400_0600",
    ]:
        experiment = EXPERIMENTS[experiment_name]

        label, risk, conf = fusion_predict(
            result,
            experiment["image_weight"],
            experiment["sensor_weight"],
            threshold,
        )

        result[experiment_name] = label
        result[f"{experiment_name}_risk"] = risk
        result[f"{experiment_name}_confidence"] = conf

    return result


def evaluate(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["healthy", "unhealthy"],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision_unhealthy": float(
            precision_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "recall_unhealthy": float(
            recall_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "f1_unhealthy": float(
            f1_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "correct": int(
            np.sum(
                np.asarray(y_true)
                == np.asarray(y_pred)
            )
        ),
        "incorrect": int(
            np.sum(
                np.asarray(y_true)
                != np.asarray(y_pred)
            )
        ),
        "tn_healthy_correct": int(tn),
        "fp_healthy_as_unhealthy": int(fp),
        "fn_unhealthy_as_healthy": int(fn),
        "tp_unhealthy_correct": int(tp),
    }


def add_disagreement_analysis(df):
    result = df.copy()

    result["image_correct"] = (
        result["image_label"] == result["ground_truth"]
    )

    result["sensor_correct"] = (
        result["sensor_label"] == result["ground_truth"]
    )

    result["image_sensor_disagree"] = (
        result["image_label"] != result["sensor_label"]
    )

    result["disagreement_case"] = "image_sensor_agree"

    mask = result["image_sensor_disagree"]

    result.loc[
        mask
        & result["image_correct"]
        & ~result["sensor_correct"],
        "disagreement_case",
    ] = "image_correct_sensor_wrong"

    result.loc[
        mask
        & ~result["image_correct"]
        & result["sensor_correct"],
        "disagreement_case",
    ] = "sensor_correct_image_wrong"

    result.loc[
        mask
        & ~result["image_correct"]
        & ~result["sensor_correct"],
        "disagreement_case",
    ] = "both_wrong"

    for experiment_name in EXPERIMENTS:
        result[f"{experiment_name}_correct"] = (
            result[experiment_name]
            == result["ground_truth"]
        )

    return result


def validate_60_40(df):
    comparable = df[
        df["stored_fusion_label"].notna()
    ].copy()

    if comparable.empty:
        return None

    matches = (
        comparable["E3_Fusion_0600_0400"]
        == comparable["stored_fusion_label"]
    )

    total = len(comparable)
    match_count = int(matches.sum())

    return {
        "total": total,
        "matches": match_count,
        "mismatches": total - match_count,
        "match_pct": match_count / total * 100.0,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, threshold = load_data()
    df = add_risks(df)
    df = build_predictions(df, threshold)
    df = add_disagreement_analysis(df)

    metric_rows = []
    confusion_rows = []

    for experiment_name, experiment in EXPERIMENTS.items():
        metrics = evaluate(
            df["ground_truth"],
            df[experiment_name],
        )

        row = {
            "experiment": experiment_name,
            "type": experiment["type"],
            "image_weight": experiment.get(
                "image_weight",
                1.0 if experiment["type"] == "image" else 0.0,
            ),
            "sensor_weight": experiment.get(
                "sensor_weight",
                1.0 if experiment["type"] == "sensor" else 0.0,
            ),
            "samples": len(df),
            "healthy_ground_truth": int(
                (df["ground_truth"] == "healthy").sum()
            ),
            "unhealthy_ground_truth": int(
                (df["ground_truth"] == "unhealthy").sum()
            ),
        }

        row.update(metrics)
        metric_rows.append(row)

        confusion_rows.append(
            {
                "experiment": experiment_name,
                "true_healthy_pred_healthy": metrics[
                    "tn_healthy_correct"
                ],
                "true_healthy_pred_unhealthy": metrics[
                    "fp_healthy_as_unhealthy"
                ],
                "true_unhealthy_pred_healthy": metrics[
                    "fn_unhealthy_as_healthy"
                ],
                "true_unhealthy_pred_unhealthy": metrics[
                    "tp_unhealthy_correct"
                ],
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    confusion_df.to_csv(
        CONFUSION_FILE,
        index=False,
    )

    df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    validation = validate_60_40(df)

    print()
    print("=" * 105)
    print(" PRELIMINARY GROUND-TRUTH MODEL EVALUATION")
    print("=" * 105)

    print(
        f"Binary labelled observations : {len(df)}"
    )
    print(
        f"Healthy ground truth         : "
        f"{int((df['ground_truth'] == 'healthy').sum())}"
    )
    print(
        f"Unhealthy ground truth       : "
        f"{int((df['ground_truth'] == 'unhealthy').sum())}"
    )
    print(
        f"Unique captures              : "
        f"{df['capture_id'].nunique()}"
    )
    print(
        f"Unique plants                : "
        f"{df['plant_id'].nunique()}"
    )
    print(
        f"Decision threshold           : {threshold:.3f}"
    )
    print()

    display = metrics_df[
        [
            "experiment",
            "accuracy",
            "balanced_accuracy",
            "precision_unhealthy",
            "recall_unhealthy",
            "f1_unhealthy",
            "correct",
            "incorrect",
        ]
    ].copy()

    display.columns = [
        "Experiment",
        "Accuracy",
        "Balanced Acc",
        "Precision(U)",
        "Recall(U)",
        "F1(U)",
        "Correct",
        "Wrong",
    ]

    print(
        display.to_string(
            index=False,
            formatters={
                "Accuracy": lambda x: f"{x:.4f}",
                "Balanced Acc": lambda x: f"{x:.4f}",
                "Precision(U)": lambda x: f"{x:.4f}",
                "Recall(U)": lambda x: f"{x:.4f}",
                "F1(U)": lambda x: f"{x:.4f}",
            },
        )
    )

    print()
    print("CONFUSION MATRIX COUNTS")

    confusion_display = metrics_df[
        [
            "experiment",
            "tn_healthy_correct",
            "fp_healthy_as_unhealthy",
            "fn_unhealthy_as_healthy",
            "tp_unhealthy_correct",
        ]
    ].copy()

    confusion_display.columns = [
        "Experiment",
        "Healthy->Healthy",
        "Healthy->Unhealthy",
        "Unhealthy->Healthy",
        "Unhealthy->Unhealthy",
    ]

    print(confusion_display.to_string(index=False))

    print()
    disagreements = df[df["image_sensor_disagree"]]

    print(
        f"Image-Sensor disagreements  : {len(disagreements)}"
    )

    if not disagreements.empty:
        counts = (
            disagreements["disagreement_case"]
            .value_counts()
        )

        for case, count in counts.items():
            print(f"  {case:<30}: {count}")

    print()

    if validation is not None:
        print("60/40 production-fusion validation")
        print(
            f"  Comparable rows : {validation['total']}"
        )
        print(
            f"  Matches         : {validation['matches']}"
        )
        print(
            f"  Mismatches      : {validation['mismatches']}"
        )
        print(
            f"  Match rate      : {validation['match_pct']:.2f}%"
        )
        print()

    print("IMPORTANT:")
    print(
        "This is a PRELIMINARY evaluation. "
        "Do not treat it as the final dissertation result "
        "while the unhealthy class remains small."
    )
    print()
    print(f"Metrics CSV     : {METRICS_FILE}")
    print(f"Predictions CSV : {PREDICTIONS_FILE}")
    print(f"Confusion CSV   : {CONFUSION_FILE}")
    print("=" * 105)
    print()


if __name__ == "__main__":
    main()
