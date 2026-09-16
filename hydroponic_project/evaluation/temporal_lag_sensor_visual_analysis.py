#!/usr/bin/env python3

# ============================================================
# TEMPORAL-LAG SENSOR -> VISUAL HEALTH EXPLORATORY ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Test whether environmental sensor risk measured BEFORE an image
# capture is more closely associated with later visible plant
# stress than the same-time sensor reading.
#
# IMPORTANT RESEARCH LIMITATIONS
# ------------------------------
# 1. Sensor values are system/capture-level, while visual labels
#    are plant-level.
# 2. The current manual dataset contains only a small number of
#    labelled capture events, so this is EXPLORATORY analysis.
# 3. Corrective interventions after an abnormal sensor reading
#    can prevent later visible stress. Therefore a weak lag
#    relationship does not prove the sensor signal is useless.
# 4. Do not use this analysis to tune production thresholds.
#
# PRIMARY UNIT OF ANALYSIS
# ------------------------
# One CAPTURE, not one plant.
#
# For every manually-labelled capture we calculate:
#     visible_unhealthy_fraction =
#       unhealthy labelled plants / all binary-labelled plants
#
# We then compare that future visual fraction with environmental
# sensor risk at:
#     0, 15, 30, 60, 120 minutes earlier
#
# We also calculate previous-window exposure summaries:
#     previous 60 minutes
#     previous 120 minutes
#
# PRIMARY ANALYSIS FILTER
# -----------------------
# Captures with at least 5 manually-labelled plants.
#
# Captures with fewer labels are still saved for supplementary
# analysis but are not used as the primary summary.
#
# ============================================================

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

MANUAL_LABEL_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)

NEW_SENSOR_MODEL = Path(
    "/hydroponic_project/models/sensor/candidates/"
    "sensor_model_package_expert_5features.joblib"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/"
    "temporal_lag_sensor_visual"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

FEATURES = [
    "pH",
    "TDS",
    "DHT_temp",
    "DHT_humidity",
    "water_temp",
]

HEALTH_THRESHOLDS = {
    "pH_min": 5.5,
    "pH_max": 6.5,
    "TDS_min": 750.0,
    "TDS_max": 1250.0,
    "temp_min": 18.0,
    "temp_max": 31.0,
    "humidity_min": 50.0,
    "humidity_max": 90.0,
    "water_temp_min": 16.0,
    "water_temp_max": 31.0,
}

LAGS_MINUTES = [
    0,
    15,
    30,
    60,
    120,
]

WINDOWS_MINUTES = [
    60,
    120,
]

# Sensor captures are expected roughly every 15 min.
# Allow up to +/- 8 minutes around the requested lag timestamp.
MATCH_TOLERANCE_MINUTES = 8

# Primary capture-level analysis should use captures with enough
# individual plants labelled to make the unhealthy fraction useful.
MIN_LABELLED_PLANTS_PRIMARY = 5


# ============================================================
# HELPERS
# ============================================================

def load_sensor_model():
    package = joblib.load(
        NEW_SENSOR_MODEL
    )

    if not isinstance(
        package,
        dict,
    ):
        raise RuntimeError(
            "Expected the new sensor package to be a dictionary."
        )

    model = package.get(
        "model"
    )

    if model is None:
        raise RuntimeError(
            "No 'model' key found in new sensor package."
        )

    package_features = list(
        package.get(
            "features",
            []
        )
    )

    if package_features != FEATURES:
        raise RuntimeError(
            f"Feature mismatch.\n"
            f"Expected: {FEATURES}\n"
            f"Package : {package_features}"
        )

    classes = list(
        model.classes_
    )

    if 0 not in classes:
        raise RuntimeError(
            f"Class 0 not found in model classes: {classes}"
        )

    unhealthy_index = classes.index(
        0
    )

    return model, unhealthy_index


def expert_warning(row):
    """
    1 = at least one expert healthy range is violated
    0 = all five measurements are within expert ranges
    """

    warning = (
        row["ph"] < HEALTH_THRESHOLDS["pH_min"]
        or row["ph"] > HEALTH_THRESHOLDS["pH_max"]
        or row["tds_ppm"] < HEALTH_THRESHOLDS["TDS_min"]
        or row["tds_ppm"] > HEALTH_THRESHOLDS["TDS_max"]
        or row["air_temp_c"] < HEALTH_THRESHOLDS["temp_min"]
        or row["air_temp_c"] > HEALTH_THRESHOLDS["temp_max"]
        or row["humidity_pct"] < HEALTH_THRESHOLDS["humidity_min"]
        or row["humidity_pct"] > HEALTH_THRESHOLDS["humidity_max"]
        or row["water_temp_c"] < HEALTH_THRESHOLDS["water_temp_min"]
        or row["water_temp_c"] > HEALTH_THRESHOLDS["water_temp_max"]
    )

    return int(
        warning
    )


def spearman_safe(x, y):
    """
    Pandas Spearman correlation.
    Returns NaN when there are too few usable rows or no variation.
    """

    temp = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()

    if len(temp) < 3:
        return np.nan

    if temp["x"].nunique() < 2:
        return np.nan

    if temp["y"].nunique() < 2:
        return np.nan

    return float(
        temp["x"].corr(
            temp["y"],
            method="spearman",
        )
    )


# ============================================================
# LOAD MANUAL VISUAL LABELS
# ============================================================

labels = pd.read_csv(
    MANUAL_LABEL_CSV
)

if "timestamp_labels" in labels.columns:
    labels["visual_timestamp"] = pd.to_datetime(
        labels["timestamp_labels"],
        errors="coerce",
    )
elif "timestamp" in labels.columns:
    labels["visual_timestamp"] = pd.to_datetime(
        labels["timestamp"],
        errors="coerce",
    )
else:
    raise RuntimeError(
        "No timestamp column found in manual-label CSV."
    )

labels["ground_truth"] = (
    labels["ground_truth"]
    .astype(str)
    .str.strip()
    .str.lower()
)

labels = labels[
    labels["ground_truth"].isin(
        [
            "healthy",
            "unhealthy",
        ]
    )
].copy()

labels = labels.dropna(
    subset=[
        "visual_timestamp",
        "capture_id",
        "plant_id",
    ]
)


# ============================================================
# AGGREGATE VISUAL LABELS TO CAPTURE LEVEL
# ============================================================

visual_capture = (
    labels
    .groupby(
        "capture_id"
    )
    .agg(
        visual_timestamp=(
            "visual_timestamp",
            "first",
        ),
        labelled_plants=(
            "plant_id",
            "size",
        ),
        healthy_plants=(
            "ground_truth",
            lambda x:
                int(
                    (
                        x == "healthy"
                    ).sum()
                ),
        ),
        unhealthy_plants=(
            "ground_truth",
            lambda x:
                int(
                    (
                        x == "unhealthy"
                    ).sum()
                ),
        ),
    )
    .reset_index()
)

visual_capture[
    "visible_unhealthy_fraction"
] = (
    visual_capture[
        "unhealthy_plants"
    ]
    /
    visual_capture[
        "labelled_plants"
    ]
)

visual_capture[
    "any_visible_unhealthy"
] = (
    visual_capture[
        "unhealthy_plants"
    ]
    > 0
).astype(
    int
)

visual_capture = visual_capture.sort_values(
    "visual_timestamp"
).reset_index(
    drop=True
)


# ============================================================
# LOAD ALL CAPTURE SENSOR RECORDS
# ============================================================

connection = sqlite3.connect(
    str(
        DB_FILE
    )
)

try:
    sensor = pd.read_sql_query(
        """
        SELECT
            capture_id,
            timestamp,
            ph,
            tds_ppm,
            air_temp_c,
            humidity_pct,
            water_temp_c
        FROM capture_events
        ORDER BY timestamp
        """,
        connection,
    )
finally:
    connection.close()

sensor[
    "timestamp"
] = pd.to_datetime(
    sensor[
        "timestamp"
    ],
    errors="coerce",
)

sensor = sensor.dropna(
    subset=[
        "timestamp",
        "ph",
        "tds_ppm",
        "air_temp_c",
        "humidity_pct",
        "water_temp_c",
    ]
).copy()

sensor = sensor.sort_values(
    "timestamp"
).reset_index(
    drop=True
)


# ============================================================
# CALCULATE NEW RF SENSOR RISK FOR EVERY CAPTURE
# ============================================================

model, unhealthy_index = (
    load_sensor_model()
)

X_sensor = pd.DataFrame(
    {
        "pH":
            sensor["ph"],
        "TDS":
            sensor["tds_ppm"],
        "DHT_temp":
            sensor["air_temp_c"],
        "DHT_humidity":
            sensor["humidity_pct"],
        "water_temp":
            sensor["water_temp_c"],
    }
)

sensor[
    "sensor_unhealthy_probability"
] = (
    model.predict_proba(
        X_sensor[
            FEATURES
        ]
    )[
        :,
        unhealthy_index
    ]
)

sensor[
    "expert_environment_warning"
] = sensor.apply(
    expert_warning,
    axis=1,
)


# ============================================================
# MATCH PRIOR SENSOR CAPTURES AT SPECIFIC LAGS
# ============================================================

analysis = visual_capture.copy()

sensor_for_match = sensor[
    [
        "timestamp",
        "capture_id",
        "sensor_unhealthy_probability",
        "expert_environment_warning",
        "ph",
        "tds_ppm",
        "air_temp_c",
        "humidity_pct",
        "water_temp_c",
    ]
].copy()

for lag_minutes in LAGS_MINUTES:

    targets = analysis[
        [
            "capture_id",
            "visual_timestamp",
        ]
    ].copy()

    targets[
        "requested_sensor_timestamp"
    ] = (
        targets[
            "visual_timestamp"
        ]
        - pd.to_timedelta(
            lag_minutes,
            unit="m",
        )
    )

    targets = targets.sort_values(
        "requested_sensor_timestamp"
    )

    matched = pd.merge_asof(
        targets,
        sensor_for_match.sort_values(
            "timestamp"
        ),
        left_on="requested_sensor_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            minutes=
                MATCH_TOLERANCE_MINUTES
        ),
        suffixes=(
            "_visual",
            "_sensor",
        ),
    )

    matched = matched.sort_values(
        "visual_timestamp"
    )

    analysis = analysis.sort_values(
        "visual_timestamp"
    ).reset_index(
        drop=True
    )

    matched = matched.reset_index(
        drop=True
    )

    prefix = f"lag_{lag_minutes}m"

    analysis[
        f"{prefix}_sensor_timestamp"
    ] = matched[
        "timestamp"
    ]

    analysis[
        f"{prefix}_sensor_capture_id"
    ] = matched[
        "capture_id_sensor"
    ]

    analysis[
        f"{prefix}_sensor_risk"
    ] = matched[
        "sensor_unhealthy_probability"
    ]

    analysis[
        f"{prefix}_expert_warning"
    ] = matched[
        "expert_environment_warning"
    ]

    analysis[
        f"{prefix}_tds"
    ] = matched[
        "tds_ppm"
    ]

    analysis[
        f"{prefix}_ph"
    ] = matched[
        "ph"
    ]


# ============================================================
# PREVIOUS-WINDOW EXPOSURE FEATURES
# ============================================================

for window_minutes in WINDOWS_MINUTES:

    risk_mean = []
    risk_max = []
    warning_fraction = []
    tds_min = []
    tds_max = []
    records = []

    for visual_time in analysis[
        "visual_timestamp"
    ]:

        start_time = (
            visual_time
            - pd.Timedelta(
                minutes=
                    window_minutes
            )
        )

        window = sensor[
            (
                sensor["timestamp"] >= start_time
            )
            &
            (
                sensor["timestamp"] <= visual_time
            )
        ]

        records.append(
            len(
                window
            )
        )

        if window.empty:
            risk_mean.append(
                np.nan
            )
            risk_max.append(
                np.nan
            )
            warning_fraction.append(
                np.nan
            )
            tds_min.append(
                np.nan
            )
            tds_max.append(
                np.nan
            )
            continue

        risk_mean.append(
            float(
                window[
                    "sensor_unhealthy_probability"
                ].mean()
            )
        )

        risk_max.append(
            float(
                window[
                    "sensor_unhealthy_probability"
                ].max()
            )
        )

        warning_fraction.append(
            float(
                window[
                    "expert_environment_warning"
                ].mean()
            )
        )

        tds_min.append(
            float(
                window[
                    "tds_ppm"
                ].min()
            )
        )

        tds_max.append(
            float(
                window[
                    "tds_ppm"
                ].max()
            )
        )

    prefix = (
        f"previous_{window_minutes}m"
    )

    analysis[
        f"{prefix}_records"
    ] = records

    analysis[
        f"{prefix}_mean_sensor_risk"
    ] = risk_mean

    analysis[
        f"{prefix}_max_sensor_risk"
    ] = risk_max

    analysis[
        f"{prefix}_expert_warning_fraction"
    ] = warning_fraction

    analysis[
        f"{prefix}_min_tds"
    ] = tds_min

    analysis[
        f"{prefix}_max_tds"
    ] = tds_max


# ============================================================
# PRIMARY VS SUPPLEMENTARY DATASETS
# ============================================================

primary = analysis[
    analysis[
        "labelled_plants"
    ]
    >= MIN_LABELLED_PLANTS_PRIMARY
].copy()


# ============================================================
# SUMMARY METRICS
# ============================================================

summary_rows = []

for dataset_name, dataset in [
    (
        "All labelled captures",
        analysis,
    ),
    (
        f"Primary captures >= {MIN_LABELLED_PLANTS_PRIMARY} plants",
        primary,
    ),
]:

    for lag_minutes in LAGS_MINUTES:

        risk_col = (
            f"lag_{lag_minutes}m_sensor_risk"
        )

        warning_col = (
            f"lag_{lag_minutes}m_expert_warning"
        )

        valid = dataset.dropna(
            subset=[
                risk_col,
                "visible_unhealthy_fraction",
            ]
        )

        risk_spearman = spearman_safe(
            valid[
                risk_col
            ],
            valid[
                "visible_unhealthy_fraction"
            ],
        )

        warning_spearman = spearman_safe(
            valid[
                warning_col
            ],
            valid[
                "visible_unhealthy_fraction"
            ],
        )

        # Exploratory capture-level AUC:
        # target = whether ANY manually labelled plant is unhealthy.
        # This is secondary only because capture count is small.
        auc = np.nan

        if (
            len(
                valid
            )
            >= 4
            and valid[
                "any_visible_unhealthy"
            ].nunique()
            == 2
        ):
            auc = float(
                roc_auc_score(
                    valid[
                        "any_visible_unhealthy"
                    ],
                    valid[
                        risk_col
                    ],
                )
            )

        summary_rows.append(
            {
                "dataset":
                    dataset_name,
                "analysis_type":
                    "single_lag",
                "lag_or_window_minutes":
                    lag_minutes,
                "matched_captures":
                    len(
                        valid
                    ),
                "sensor_risk_vs_visible_fraction_spearman":
                    risk_spearman,
                "expert_warning_vs_visible_fraction_spearman":
                    warning_spearman,
                "sensor_risk_auc_any_visible_unhealthy":
                    auc,
            }
        )

    for window_minutes in WINDOWS_MINUTES:

        mean_col = (
            f"previous_{window_minutes}m_mean_sensor_risk"
        )

        max_col = (
            f"previous_{window_minutes}m_max_sensor_risk"
        )

        warn_col = (
            f"previous_{window_minutes}m_expert_warning_fraction"
        )

        valid = dataset.dropna(
            subset=[
                mean_col,
                "visible_unhealthy_fraction",
            ]
        )

        summary_rows.append(
            {
                "dataset":
                    dataset_name,
                "analysis_type":
                    "exposure_window_mean",
                "lag_or_window_minutes":
                    window_minutes,
                "matched_captures":
                    len(
                        valid
                    ),
                "sensor_risk_vs_visible_fraction_spearman":
                    spearman_safe(
                        valid[
                            mean_col
                        ],
                        valid[
                            "visible_unhealthy_fraction"
                        ],
                    ),
                "expert_warning_vs_visible_fraction_spearman":
                    spearman_safe(
                        valid[
                            warn_col
                        ],
                        valid[
                            "visible_unhealthy_fraction"
                        ],
                    ),
                "sensor_risk_auc_any_visible_unhealthy":
                    np.nan,
            }
        )

        summary_rows.append(
            {
                "dataset":
                    dataset_name,
                "analysis_type":
                    "exposure_window_max",
                "lag_or_window_minutes":
                    window_minutes,
                "matched_captures":
                    len(
                        valid
                    ),
                "sensor_risk_vs_visible_fraction_spearman":
                    spearman_safe(
                        valid[
                            max_col
                        ],
                        valid[
                            "visible_unhealthy_fraction"
                        ],
                    ),
                "expert_warning_vs_visible_fraction_spearman":
                    np.nan,
                "sensor_risk_auc_any_visible_unhealthy":
                    np.nan,
            }
        )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

analysis.to_csv(
    OUTPUT_DIR
    / "temporal_lag_capture_analysis_all.csv",
    index=False,
)

primary.to_csv(
    OUTPUT_DIR
    / "temporal_lag_capture_analysis_primary.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR
    / "temporal_lag_summary.csv",
    index=False,
)

sensor.to_csv(
    OUTPUT_DIR
    / "all_capture_sensor_risk.csv",
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 95)
print(
    " TEMPORAL-LAG SENSOR -> VISUAL HEALTH EXPLORATORY ANALYSIS"
)
print("=" * 95)

print(
    "Binary manual plant labels :",
    len(
        labels
    )
)

print(
    "Labelled visual captures   :",
    len(
        analysis
    )
)

print(
    "Primary captures           :",
    len(
        primary
    ),
    f"(>= {MIN_LABELLED_PLANTS_PRIMARY} labelled plants)"
)

print(
    "All sensor captures loaded :",
    len(
        sensor
    )
)

print()

print(
    "Primary capture-level visual data:"
)

print(
    primary[
        [
            "capture_id",
            "visual_timestamp",
            "labelled_plants",
            "healthy_plants",
            "unhealthy_plants",
            "visible_unhealthy_fraction",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 95)
print(
    " PRIMARY TEMPORAL-LAG SUMMARY"
)
print("=" * 95)

primary_summary = summary[
    summary[
        "dataset"
    ].str.startswith(
        "Primary"
    )
]

print(
    primary_summary[
        [
            "analysis_type",
            "lag_or_window_minutes",
            "matched_captures",
            "sensor_risk_vs_visible_fraction_spearman",
            "expert_warning_vs_visible_fraction_spearman",
            "sensor_risk_auc_any_visible_unhealthy",
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
    "This is an exploratory temporal sensitivity analysis, "
    "not proof of a biological response delay."
)

print(
    "The current manually-labelled dataset contains only a "
    "small number of capture events, and corrective actions "
    "after abnormal pH/TDS readings may prevent later visible stress."
)

print()
print(
    "Outputs:"
)

print(
    OUTPUT_DIR
    / "temporal_lag_capture_analysis_all.csv"
)

print(
    OUTPUT_DIR
    / "temporal_lag_capture_analysis_primary.csv"
)

print(
    OUTPUT_DIR
    / "temporal_lag_summary.csv"
)

print("=" * 95)
