#!/usr/bin/env python3

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

DB_FILE = Path("/hydroponic_project/database/hydroponic.db")
LIVE_LABEL_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)
OLD_MODEL_PACKAGE = Path(
    "/hydroponic_project/models/sensor/sensor_model_package.joblib"
)
NEW_MODEL_PACKAGE = Path(
    "/hydroponic_project/models/sensor/candidates/"
    "sensor_model_package_expert_5features.joblib"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/"
    "sensor_live_ground_truth_comparison"
)

PREDICTIONS_CSV = OUTPUT_DIR / "sensor_live_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "sensor_live_model_metrics.csv"
CONFUSION_CSV = OUTPUT_DIR / "sensor_live_confusion_counts.csv"
CAPTURE_SUMMARY_CSV = OUTPUT_DIR / "sensor_live_capture_summary.csv"

OLD_EXPECTED_FEATURES = [
    "pH",
    "TDS",
    "water_level",
    "DHT_temp",
    "DHT_humidity",
    "water_temp",
]

NEW_EXPECTED_FEATURES = [
    "pH",
    "TDS",
    "DHT_temp",
    "DHT_humidity",
    "water_temp",
]

OLD_THRESHOLD = 0.50
NEW_REFERENCE_THRESHOLD = 0.50


def encode_old_water_level(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    if text in {"OK", "NORMAL", "HIGH", "1", "TRUE"}:
        return 1.0

    if text in {"LOW", "0", "FALSE"}:
        return 0.0

    raise ValueError(
        f"Unknown water_level value in capture_events: {value}"
    )


def load_manual_labels():
    if not LIVE_LABEL_CSV.exists():
        raise FileNotFoundError(
            f"Manual-label CSV not found:\n{LIVE_LABEL_CSV}"
        )

    df = pd.read_csv(LIVE_LABEL_CSV)

    required = {"capture_id", "plant_id", "ground_truth"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["ground_truth"] = (
        df["ground_truth"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = (
        df[
            df["ground_truth"].isin(
                ["healthy", "unhealthy"]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if df.empty:
        raise RuntimeError(
            "No binary Healthy/Unhealthy labels found."
        )

    return df


def load_capture_sensor_data():
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"SQLite database not found:\n{DB_FILE}"
        )

    connection = sqlite3.connect(str(DB_FILE))

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
            connection,
        )
    finally:
        connection.close()

    sensor_df["old_water_level_numeric"] = (
        sensor_df["water_level"].apply(
            encode_old_water_level
        )
    )

    return sensor_df


def unpack_model_package(
    package_path,
    expected_features,
    package_label,
):
    if not package_path.exists():
        raise FileNotFoundError(
            f"{package_label} package not found:\n"
            f"{package_path}"
        )

    package = joblib.load(package_path)

    if hasattr(package, "predict_proba"):
        model = package
        feature_names = None
        saved_threshold = None
        package_dict = None

    elif isinstance(package, dict):
        package_dict = package
        model = None

        for key in [
            "model",
            "classifier",
            "rf_model",
            "random_forest",
        ]:
            candidate = package.get(key)

            if (
                candidate is not None
                and hasattr(candidate, "predict_proba")
            ):
                model = candidate
                break

        if model is None:
            raise RuntimeError(
                f"Could not locate model in {package_label} package."
            )

        feature_names = (
            package.get("features")
            or package.get("feature_names")
            or package.get("feature_columns")
        )

        saved_threshold = package.get(
            "decision_threshold_unhealthy_risk"
        )

    else:
        raise RuntimeError(
            f"Unsupported package format for {package_label}: "
            f"{type(package)}"
        )

    if feature_names is not None:
        feature_names = list(feature_names)

        if feature_names != expected_features:
            raise RuntimeError(
                f"{package_label} feature mismatch.\n"
                f"Expected: {expected_features}\n"
                f"Package : {feature_names}"
            )

    classes = list(model.classes_)

    if 0 not in classes or 1 not in classes:
        raise RuntimeError(
            f"{package_label} has unexpected classes: {classes}"
        )

    return {
        "package": package_dict,
        "model": model,
        "features": (
            feature_names
            if feature_names is not None
            else expected_features
        ),
        "saved_threshold": saved_threshold,
        "classes": classes,
        "unhealthy_class_index": classes.index(0),
    }


def create_feature_matrices(df):
    old_X = pd.DataFrame(
        {
            "pH": df["ph"],
            "TDS": df["tds_ppm"],
            "water_level": df["old_water_level_numeric"],
            "DHT_temp": df["air_temp_c"],
            "DHT_humidity": df["humidity_pct"],
            "water_temp": df["water_temp_c"],
        }
    )

    new_X = pd.DataFrame(
        {
            "pH": df["ph"],
            "TDS": df["tds_ppm"],
            "DHT_temp": df["air_temp_c"],
            "DHT_humidity": df["humidity_pct"],
            "water_temp": df["water_temp_c"],
        }
    )

    if old_X.isna().any().any():
        raise RuntimeError(
            "Missing values exist in OLD model features."
        )

    if new_X.isna().any().any():
        raise RuntimeError(
            "Missing values exist in NEW model features."
        )

    return (
        old_X[OLD_EXPECTED_FEATURES],
        new_X[NEW_EXPECTED_FEATURES],
    )


def get_unhealthy_probability(
    model_info,
    X,
):
    probabilities = model_info["model"].predict_proba(X)

    return probabilities[
        :,
        model_info["unhealthy_class_index"],
    ]


def evaluate_predictions(
    experiment,
    y_true,
    unhealthy_probability,
    threshold,
):
    unhealthy_probability = np.asarray(
        unhealthy_probability,
        dtype=float,
    )

    y_pred = np.where(
        unhealthy_probability >= threshold,
        "unhealthy",
        "healthy",
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["healthy", "unhealthy"],
    )

    tn, fp, fn, tp = cm.ravel()

    y_true_binary = np.where(
        np.asarray(y_true) == "unhealthy",
        1,
        0,
    )

    metrics = {
        "experiment": experiment,
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "unhealthy_precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "unhealthy_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "unhealthy_f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label="unhealthy",
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true_binary,
                unhealthy_probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true_binary,
                unhealthy_probability,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true_binary,
                unhealthy_probability,
            )
        ),
        "healthy_correct_tn": int(tn),
        "healthy_as_unhealthy_fp": int(fp),
        "unhealthy_as_healthy_fn": int(fn),
        "unhealthy_correct_tp": int(tp),
    }

    return metrics, y_pred


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_df = load_manual_labels()
    sensor_df = load_capture_sensor_data()

    df = label_df.merge(
        sensor_df,
        on="capture_id",
        how="left",
        validate="many_to_one",
        suffixes=("_labels", "_sensor"),
        indicator=True,
    )

    unmatched = df["_merge"] != "both"

    if unmatched.any():
        missing_ids = (
            df.loc[
                unmatched,
                "capture_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Some labelled observations could not be matched "
            f"to capture_events: {missing_ids}"
        )

    df = df.drop(columns=["_merge"])

    old_info = unpack_model_package(
        OLD_MODEL_PACKAGE,
        OLD_EXPECTED_FEATURES,
        "OLD SENSOR MODEL",
    )

    new_info = unpack_model_package(
        NEW_MODEL_PACKAGE,
        NEW_EXPECTED_FEATURES,
        "NEW 5-FEATURE SENSOR MODEL",
    )

    new_selected_threshold = new_info["saved_threshold"]

    if new_selected_threshold is None:
        raise RuntimeError(
            "New package does not contain "
            "'decision_threshold_unhealthy_risk'."
        )

    new_selected_threshold = float(
        new_selected_threshold
    )

    old_X, new_X = create_feature_matrices(df)

    old_probability = get_unhealthy_probability(
        old_info,
        old_X,
    )

    new_probability = get_unhealthy_probability(
        new_info,
        new_X,
    )

    y_true = df["ground_truth"].to_numpy()

    old_metrics, old_pred = evaluate_predictions(
        "Old_Sensor_RF_6Feature_Threshold_0.50",
        y_true,
        old_probability,
        OLD_THRESHOLD,
    )

    new_050_metrics, new_050_pred = evaluate_predictions(
        "New_Sensor_RF_5Feature_Threshold_0.50",
        y_true,
        new_probability,
        NEW_REFERENCE_THRESHOLD,
    )

    new_selected_metrics, new_selected_pred = (
        evaluate_predictions(
            (
                "New_Sensor_RF_5Feature_"
                f"Selected_Threshold_{new_selected_threshold:.2f}"
            ),
            y_true,
            new_probability,
            new_selected_threshold,
        )
    )

    metrics_df = pd.DataFrame(
        [
            old_metrics,
            new_050_metrics,
            new_selected_metrics,
        ]
    )

    df["old_sensor_unhealthy_probability"] = (
        old_probability
    )

    df["old_sensor_prediction_050"] = (
        old_pred
    )

    df["new_sensor_unhealthy_probability"] = (
        new_probability
    )

    df["new_sensor_prediction_050"] = (
        new_050_pred
    )

    df["new_sensor_selected_threshold"] = (
        new_selected_threshold
    )

    df["new_sensor_prediction_selected"] = (
        new_selected_pred
    )

    df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_CSV,
        index=False,
    )

    metrics_df[
        [
            "experiment",
            "healthy_correct_tn",
            "healthy_as_unhealthy_fp",
            "unhealthy_as_healthy_fn",
            "unhealthy_correct_tp",
        ]
    ].to_csv(
        CONFUSION_CSV,
        index=False,
    )

    capture_summary = (
        df
        .groupby("capture_id")
        .agg(
            labelled_plants=(
                "plant_id",
                "size",
            ),
            healthy_ground_truth=(
                "ground_truth",
                lambda x:
                    int((x == "healthy").sum()),
            ),
            unhealthy_ground_truth=(
                "ground_truth",
                lambda x:
                    int((x == "unhealthy").sum()),
            ),
            old_sensor_unhealthy_probability=(
                "old_sensor_unhealthy_probability",
                "first",
            ),
            new_sensor_unhealthy_probability=(
                "new_sensor_unhealthy_probability",
                "first",
            ),
            old_sensor_prediction_050=(
                "old_sensor_prediction_050",
                "first",
            ),
            new_sensor_prediction_selected=(
                "new_sensor_prediction_selected",
                "first",
            ),
        )
        .reset_index()
    )

    capture_summary.to_csv(
        CAPTURE_SUMMARY_CSV,
        index=False,
    )

    print()
    print("=" * 108)
    print(
        " OLD vs NEW SENSOR MODEL - LIVE MANUAL GROUND-TRUTH EVALUATION"
    )
    print("=" * 108)

    print(
        "Binary labelled observations :",
        len(df)
    )
    print(
        "Healthy ground truth         :",
        int((df["ground_truth"] == "healthy").sum())
    )
    print(
        "Unhealthy ground truth       :",
        int((df["ground_truth"] == "unhealthy").sum())
    )
    print(
        "Unique captures              :",
        df["capture_id"].nunique()
    )

    print()
    print(
        "Old model features           :",
        OLD_EXPECTED_FEATURES
    )
    print(
        "New model features           :",
        NEW_EXPECTED_FEATURES
    )
    print(
        "Old threshold                :",
        f"{OLD_THRESHOLD:.2f}"
    )
    print(
        "New validation threshold     :",
        f"{new_selected_threshold:.2f}"
    )

    print()
    print("=" * 108)
    print(" LIVE SENSOR MODEL COMPARISON")
    print("=" * 108)

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
        metrics_df[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 108)
    print(
        " CHANGE: OLD 0.50 -> NEW SELECTED THRESHOLD"
    )
    print("=" * 108)

    print(
        "Accuracy change             : "
        f"{(
            new_selected_metrics['accuracy']
            - old_metrics['accuracy']
        ) * 100:+.2f} percentage points"
    )

    print(
        "Balanced Accuracy change    : "
        f"{(
            new_selected_metrics['balanced_accuracy']
            - old_metrics['balanced_accuracy']
        ) * 100:+.2f} percentage points"
    )

    print(
        "Macro-F1 change             : "
        f"{(
            new_selected_metrics['macro_f1']
            - old_metrics['macro_f1']
        ) * 100:+.2f} percentage points"
    )

    print(
        "Unhealthy Recall change     : "
        f"{(
            new_selected_metrics['unhealthy_recall']
            - old_metrics['unhealthy_recall']
        ) * 100:+.2f} percentage points"
    )

    print(
        "Unhealthy F1 change         : "
        f"{(
            new_selected_metrics['unhealthy_f1']
            - old_metrics['unhealthy_f1']
        ) * 100:+.2f} percentage points"
    )

    print()
    print("IMPORTANT:")
    print(
        "Sensor values are capture-level, so all plants "
        "within one capture share the same sensor prediction."
    )
    print(
        "This live evaluation tests how well the expert-derived "
        "environmental risk score corresponds to the manually "
        "assigned visible plant-health labels."
    )

    print()
    print("Predictions CSV:")
    print(PREDICTIONS_CSV)
    print("Metrics CSV:")
    print(METRICS_CSV)
    print("Capture summary CSV:")
    print(CAPTURE_SUMMARY_CSV)
    print("=" * 108)


if __name__ == "__main__":
    main()
