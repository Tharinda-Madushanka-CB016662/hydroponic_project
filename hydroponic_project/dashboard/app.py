import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hydroponic Edge-AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path("/hydroponic_project/database/hydroponic.db")

# Benchmark results produced by compare_image_models.py.
MODEL_BENCHMARK_FILE = Path(
    "/hydroponic_project/evaluation/"
    "image_model_benchmark_combined_stress_clean.csv"
)


# ============================================================
# REFERENCE / QUALITY LIMITS
# ============================================================

PH_MIN = 5.5
PH_MAX = 6.5

TDS_MIN = 750
TDS_MAX = 1250

AIR_TEMP_MIN = 18
AIR_TEMP_MAX = 31

HUMIDITY_MIN = 50
HUMIDITY_MAX = 90

WATER_TEMP_MIN = 16
WATER_TEMP_MAX = 31

MIN_IMAGE_LUX = 10.0

IMAGE_START_HOUR = 6
IMAGE_END_HOUR = 17


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f4f7fb;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e7ecf2;
    }

    .sidebar-brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #14233a;
        margin-bottom: 2px;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #748198;
        margin-bottom: 14px;
    }

    .online-box {
        background: #eaf8ef;
        border: 1px solid #ccebd8;
        color: #167447;
        padding: 10px 12px;
        border-radius: 10px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0fbf4 100%);
        border: 1px solid #e2ebe5;
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 7px 24px rgba(20, 45, 70, 0.05);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #14233a;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #68778c;
        font-size: 14px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 13px;
        background: #e8f8ef;
        color: #147745;
        border-radius: 18px;
        padding: 6px 11px;
        font-size: 12px;
        font-weight: 700;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e6ebf2;
        border-radius: 15px;
        min-height: 126px;
        padding: 17px 20px;
        box-shadow: 0 6px 18px rgba(30, 45, 70, 0.05);
    }

    .metric-blue {
        border-top: 4px solid #2196f3;
    }

    .metric-green {
        border-top: 4px solid #2fc36c;
    }

    .metric-orange {
        border-top: 4px solid #ff8b2c;
    }

    .metric-purple {
        border-top: 4px solid #8b5cf6;
    }

    .metric-title {
        font-size: 13px;
        font-weight: 700;
        color: #66758b;
    }

    .metric-value {
        margin-top: 6px;
        margin-bottom: 1px;
        font-size: 30px;
        font-weight: 800;
        color: #14233a;
    }

    .metric-caption {
        font-size: 11px;
        color: #8a97a8;
    }

    .sensor-card {
        background: #ffffff;
        border: 1px solid #e5eaf1;
        border-radius: 13px;
        min-height: 103px;
        padding: 15px 17px;
        box-shadow: 0 4px 15px rgba(30, 45, 70, 0.035);
    }

    .sensor-title {
        font-size: 12px;
        font-weight: 700;
        color: #748197;
    }

    .sensor-value {
        margin-top: 7px;
        font-size: 23px;
        font-weight: 800;
        color: #152238;
    }

    .sensor-reference {
        margin-top: 3px;
        font-size: 10px;
        color: #98a3b3;
    }

    .panel-title {
        font-size: 20px;
        font-weight: 800;
        color: #152238;
        margin-bottom: 2px;
    }

    .panel-subtitle {
        color: #8090a4;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .system-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 2px;
        border-bottom: 1px solid #edf0f5;
        font-size: 13px;
    }

    .ready-badge {
        background: #e8f8ef;
        color: #16824a;
        border-radius: 16px;
        padding: 4px 9px;
        font-weight: 700;
        font-size: 10px;
    }

    .disabled-badge {
        background: #fff3df;
        color: #a7620d;
        border-radius: 16px;
        padding: 4px 9px;
        font-weight: 700;
        font-size: 10px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        border-color: #e1e7ef;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(25, 45, 75, 0.035);
    }

    .footer {
        text-align: center;
        color: #8794a7;
        font-size: 11px;
        margin-top: 36px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_connection():
    # Every Streamlit rerun opens fresh SQLite connections.
    # Enable foreign-key enforcement on each connection so the
    # evaluation-label table stays consistent with plant_samples.
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def check_database():
    if not DB_FILE.exists():
        st.error(f"Database not found: {DB_FILE}")
        st.stop()

    try:
        conn = get_connection()
        conn.execute("SELECT 1 FROM capture_events LIMIT 1")
        conn.close()

    except Exception as error:
        st.error(f"Database error: {error}")
        st.stop()


def ensure_final_decision_schema():
    """
    Add the final complementary-decision columns if they are missing.

    Historical classical-fusion values are preserved. New captures use
    decision_state plus explicit image/sensor risk probabilities.
    """

    conn = get_connection()

    try:
        plant_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(plant_samples)"
            ).fetchall()
        }

        for name, sql_type in {
            "image_unhealthy_probability": "REAL",
            "sensor_stress_probability": "REAL",
            "decision_state": "TEXT",
        }.items():
            if name not in plant_columns:
                conn.execute(
                    f"ALTER TABLE plant_samples ADD COLUMN {name} {sql_type}"
                )

        capture_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(capture_events)"
            ).fetchall()
        }

        if "environmental_warning" not in capture_columns:
            conn.execute(
                "ALTER TABLE capture_events ADD COLUMN environmental_warning INTEGER"
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# DASHBOARD / INFERENCE SETTINGS
# ============================================================
#
# These settings are stored in SQLite instead of being hard-coded
# only in the dashboard. run_inference.py reads the same values,
# therefore changing them here changes FUTURE inference behaviour.
# Historical predictions are never silently rewritten.
# ============================================================

DEFAULT_SETTINGS = {
    "min_image_lux": "10.0",
    "image_start_hour": "6",
    "image_end_hour": "17",
    "image_unhealthy_threshold": "0.42",
    "sensor_stress_threshold": "0.48",
    "ph_min": "5.5",
    "ph_max": "6.5",
    "tds_min": "750",
    "tds_max": "1250",
    "air_temp_min": "18",
    "air_temp_max": "31",
    "humidity_min": "50",
    "humidity_max": "90",
    "water_temp_min": "16",
    "water_temp_max": "31",
}


def ensure_system_settings():
    """Create the shared system_settings table and default values."""

    conn = get_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL,
                description TEXT,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            )
            """
        )

        descriptions = {
            "min_image_lux": "Minimum camera Lux required for image AI",
            "image_start_hour": "Image inference start hour (24-hour clock)",
            "image_end_hour": "Image inference end hour, exclusive",
            "image_unhealthy_threshold": "Final MobileNetV2 unhealthy probability threshold",
            "sensor_stress_threshold": "Final 5-feature RF environmental-stress probability threshold",
            "ph_min": "Expert-confirmed minimum pH",
            "ph_max": "Expert-confirmed maximum pH",
            "tds_min": "Expert-confirmed minimum TDS ppm",
            "tds_max": "Expert-confirmed maximum TDS ppm",
            "air_temp_min": "Expert-confirmed minimum air temperature C",
            "air_temp_max": "Expert-confirmed maximum air temperature C",
            "humidity_min": "Expert-confirmed minimum humidity percent",
            "humidity_max": "Expert-confirmed maximum humidity percent",
            "water_temp_min": "Expert-confirmed minimum water temperature C",
            "water_temp_max": "Expert-confirmed maximum water temperature C",
        }

        for key, value in DEFAULT_SETTINGS.items():
            cursor.execute(
                """
                INSERT OR IGNORE INTO system_settings (
                    setting_key,
                    setting_value,
                    description
                )
                VALUES (?, ?, ?)
                """,
                (
                    key,
                    value,
                    descriptions.get(key),
                ),
            )

        conn.commit()

    finally:
        conn.close()


def load_system_settings():
    """Load settings and convert them to useful Python numeric types."""

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT setting_key, setting_value
            FROM system_settings
            """
        ).fetchall()

    finally:
        conn.close()

    raw = {
        row["setting_key"]: row["setting_value"]
        for row in rows
    }

    # Defaults protect the application if a key is accidentally removed.
    for key, value in DEFAULT_SETTINGS.items():
        raw.setdefault(key, value)

    return {
        "min_image_lux": float(raw["min_image_lux"]),
        "image_start_hour": int(float(raw["image_start_hour"])),
        "image_end_hour": int(float(raw["image_end_hour"])),
        "image_unhealthy_threshold": float(raw["image_unhealthy_threshold"]),
        "sensor_stress_threshold": float(raw["sensor_stress_threshold"]),
        "ph_min": float(raw["ph_min"]),
        "ph_max": float(raw["ph_max"]),
        "tds_min": float(raw["tds_min"]),
        "tds_max": float(raw["tds_max"]),
        "air_temp_min": float(raw["air_temp_min"]),
        "air_temp_max": float(raw["air_temp_max"]),
        "humidity_min": float(raw["humidity_min"]),
        "humidity_max": float(raw["humidity_max"]),
        "water_temp_min": float(raw["water_temp_min"]),
        "water_temp_max": float(raw["water_temp_max"]),
    }


def save_system_settings(values):
    """Persist settings from the dashboard Settings page."""

    conn = get_connection()

    try:
        cursor = conn.cursor()

        for key, value in values.items():
            cursor.execute(
                """
                UPDATE system_settings
                SET
                    setting_value = ?,
                    updated_at = datetime('now','localtime')
                WHERE setting_key = ?
                """,
                (
                    str(value),
                    key,
                ),
            )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# PLANT POSITION MANAGEMENT
# ============================================================


def get_plant_positions():
    """Load current P01-P12 active/inactive configuration."""

    conn = get_connection()

    try:
        return pd.read_sql_query(
            """
            SELECT
                plant_id,
                active,
                plant_name,
                planted_date,
                notes,
                updated_at
            FROM plant_positions
            ORDER BY plant_id
            """,
            conn,
        )

    finally:
        conn.close()


def update_plant_positions(activity_map):
    """
    Update current occupancy.

    collect_sample.py reads plant_positions when a new capture is made.
    Therefore this changes FUTURE plant_samples rows only. Existing
    historical plant_samples.plant_active values remain unchanged.
    """

    conn = get_connection()

    try:
        cursor = conn.cursor()

        for plant_id, active in activity_map.items():
            cursor.execute(
                """
                UPDATE plant_positions
                SET
                    active = ?,
                    updated_at = datetime('now','localtime')
                WHERE plant_id = ?
                """,
                (
                    int(bool(active)),
                    plant_id,
                ),
            )

        conn.commit()

    finally:
        conn.close()



# ============================================================
# GROUND-TRUTH / EVALUATION LABELLING
# ============================================================
#
# These functions support blinded human/expert labelling.
#
# Important research rule:
#   The human label should be entered BEFORE AI predictions are
#   shown. This reduces anchoring bias from seeing a 99% model
#   prediction before deciding whether the plant looks healthy.
#
# manual_labels is completely separate from model predictions.
# The production inference pipeline never reads this table.
# ============================================================


def ensure_manual_labels_table():
    """
    Create the independent evaluation-label table if it does
    not already exist.

    Labels:
        healthy
        unhealthy
        uncertain

    'uncertain' is retained rather than forcing a reviewer to
    invent a binary label for ambiguous images.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS manual_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                capture_id TEXT NOT NULL,
                plant_id TEXT NOT NULL,

                ground_truth_label TEXT NOT NULL
                    CHECK (
                        ground_truth_label IN (
                            'healthy',
                            'unhealthy',
                            'uncertain'
                        )
                    ),

                validation_method TEXT
                    DEFAULT 'manual_visual',

                reviewer_role TEXT,

                notes TEXT,

                labeled_at TEXT
                    DEFAULT (datetime('now','localtime')),

                updated_at TEXT
                    DEFAULT (datetime('now','localtime')),

                UNIQUE (
                    capture_id,
                    plant_id
                ),

                FOREIGN KEY (
                    capture_id,
                    plant_id
                )
                REFERENCES plant_samples (
                    capture_id,
                    plant_id
                )
                ON DELETE CASCADE
            )
            """
        )

        conn.commit()

    finally:
        conn.close()


def get_evaluation_candidates():
    """Load active observations suitable for blinded visible-health labelling.

    Classical fusion is NOT required. The final deployed architecture keeps
    image-visible health and sensor environmental risk as complementary signals.
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT
                ce.capture_id,
                ce.timestamp,
                ce.full_image_path,
                ce.camera_lux,

                ps.plant_id,
                ps.plant_active,
                ps.crop_path,
                ps.model_image_path,

                ps.image_prediction,
                ps.image_confidence,
                ps.image_unhealthy_probability,

                ps.sensor_prediction,
                ps.sensor_confidence,
                ps.sensor_stress_probability,

                ps.decision_state,

                -- Preserved legacy research fields for historical audit.
                ps.fusion_prediction,
                ps.fusion_confidence,
                ps.final_status,

                ml.ground_truth_label,
                ml.validation_method,
                ml.reviewer_role,
                ml.notes,
                ml.labeled_at,
                ml.updated_at

            FROM plant_samples ps

            INNER JOIN capture_events ce
                ON ce.capture_id = ps.capture_id

            LEFT JOIN manual_labels ml
                ON ml.capture_id = ps.capture_id
               AND ml.plant_id = ps.plant_id

            WHERE
                ps.plant_active = 1

                AND LOWER(
                    TRIM(COALESCE(ps.image_prediction, ''))
                ) IN ('healthy', 'unhealthy')

                AND ps.sensor_prediction IS NOT NULL

            ORDER BY ce.timestamp, ps.plant_id
            """,
            conn,
        )

    finally:
        conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    return df

def save_manual_label(
    capture_id,
    plant_id,
    ground_truth_label,
    validation_method,
    reviewer_role,
    notes,
):
    """
    Insert or update one independent ground-truth label.

    ON CONFLICT allows the reviewer to correct a label later
    without creating duplicate rows.
    """

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO manual_labels (
                capture_id,
                plant_id,
                ground_truth_label,
                validation_method,
                reviewer_role,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?)

            ON CONFLICT (
                capture_id,
                plant_id
            )

            DO UPDATE SET
                ground_truth_label = excluded.ground_truth_label,
                validation_method = excluded.validation_method,
                reviewer_role = excluded.reviewer_role,
                notes = excluded.notes,
                updated_at = datetime('now','localtime')
            """,
            (
                capture_id,
                plant_id,
                ground_truth_label,
                validation_method,
                reviewer_role.strip() if reviewer_role else None,
                notes.strip() if notes else None,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_manual_labels_export():
    """Export manual labels with final and legacy model evidence."""

    conn = get_connection()

    try:
        return pd.read_sql_query(
            """
            SELECT
                ce.timestamp,
                ce.capture_id,
                ps.plant_id,

                ml.ground_truth_label,
                ml.validation_method,
                ml.reviewer_role,
                ml.notes,
                ml.labeled_at,
                ml.updated_at,

                ps.image_prediction,
                ps.image_confidence,
                ps.image_unhealthy_probability,

                ps.sensor_prediction,
                ps.sensor_confidence,
                ps.sensor_stress_probability,

                ps.decision_state,

                ps.fusion_prediction AS legacy_fusion_prediction,
                ps.fusion_confidence AS legacy_fusion_confidence,
                ps.final_status,

                ce.ph,
                ce.ec_ms_cm,
                ce.tds_ppm,
                ce.water_temp_c,
                ce.air_temp_c,
                ce.humidity_pct,
                ce.water_level,
                ce.camera_lux,
                ce.environmental_warning

            FROM manual_labels ml

            INNER JOIN plant_samples ps
                ON ps.capture_id = ml.capture_id
               AND ps.plant_id = ml.plant_id

            INNER JOIN capture_events ce
                ON ce.capture_id = ml.capture_id

            ORDER BY ce.timestamp, ps.plant_id
            """,
            conn,
        )

    finally:
        conn.close()

def normalise_evaluation_prediction(value):
    """
    Convert model terminology into the binary health classes
    used for simple post-label comparison.

    Sensor 'stress' terminology is mapped to the unhealthy/risk
    side for this comparison, while the dissertation should
    still explain that sensor AI estimates environmental risk
    rather than visual disease diagnosis.
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


def apply_hourly_representative_sampling(df):
    """
    Reduce highly correlated 15-minute captures by selecting
    one representative capture per clock hour per day.

    The capture closest to the start of each hour is selected,
    then ALL eligible plants from that chosen capture remain
    available for labelling.

    Example:
        10:00, 10:15, 10:30, 10:45
            -> normally keep 10:00

    If 10:00 is absent, the closest available capture in that
    hour is selected.
    """

    if df.empty:
        return df.copy()

    work = df.dropna(
        subset=["timestamp"]
    ).copy()

    if work.empty:
        return work

    captures = (
        work[
            [
                "capture_id",
                "timestamp",
            ]
        ]
        .drop_duplicates(
            subset=["capture_id"]
        )
        .copy()
    )

    captures["hour_start"] = (
        captures["timestamp"]
        .dt.floor("h")
    )

    captures["distance_seconds"] = (
        captures["timestamp"]
        - captures["hour_start"]
    ).abs().dt.total_seconds()

    selected_capture_ids = (
        captures
        .sort_values(
            [
                "hour_start",
                "distance_seconds",
                "timestamp",
            ]
        )
        .groupby(
            "hour_start",
            as_index=False,
        )
        .first()[
            "capture_id"
        ]
        .tolist()
    )

    return work[
        work["capture_id"].isin(
            selected_capture_ids
        )
    ].copy()


def render_post_label_ai_comparison(row):
    """Reveal model outputs only after an independent visible-health label exists."""

    truth = str(row.get("ground_truth_label", "")).strip().lower()

    st.markdown("### AI Results — Revealed After Labelling")

    if truth == "uncertain":
        st.info(
            "Ground truth is uncertain. Exclude this sample from the main "
            "binary accuracy/F1 calculation."
        )

    # Image AI predicts the same visible-health target as the manual label.
    image_prediction = row.get("image_prediction")
    image_conf = confidence(row.get("image_confidence"))
    image_normalised = normalise_evaluation_prediction(image_prediction)

    with st.container(border=True):
        st.markdown("**📷 Image AI — Visible Plant Condition**")
        st.write(
            f"{image_prediction} — {image_conf}"
            if image_conf else image_prediction
        )

        if truth in {"healthy", "unhealthy"} and image_normalised is not None:
            if image_normalised == truth:
                st.caption("✅ agrees with visible ground truth")
            else:
                st.caption("❌ disagrees with visible ground truth")
        else:
            st.caption("not binary-scored because the ground truth is uncertain")

    # Sensor AI is deliberately not scored as if it were a plant-image label.
    with st.container(border=True):
        st.markdown("**🌡️ Sensor AI — Environmental Risk**")
        sensor_prediction = row.get("sensor_prediction") or "Not evaluated"
        sensor_conf = confidence(row.get("sensor_confidence"))
        st.write(
            f"{sensor_prediction} — {sensor_conf}"
            if sensor_conf else sensor_prediction
        )
        st.caption(
            "Supplementary context only: this branch predicts current shared "
            "environmental suitability, not the same-time visual appearance of "
            "an individual plant."
        )

    decision_state = row.get("decision_state")
    if decision_state and not pd.isna(decision_state):
        with st.container(border=True):
            st.markdown("**🧠 Complementary Decision State**")
            st.write(str(decision_state))
            st.caption(
                "This decision preserves both visible-health and environmental "
                "signals and is not evaluated as a separate binary classifier."
            )

    # Historical captures may contain the retired classical fusion result.
    legacy_fusion = row.get("fusion_prediction")
    if legacy_fusion and not pd.isna(legacy_fusion):
        with st.expander("Legacy classical fusion result (research history)"):
            legacy_conf = confidence(row.get("fusion_confidence"))
            st.write(
                f"{legacy_fusion} — {legacy_conf}"
                if legacy_conf else legacy_fusion
            )

# ============================================================
# CAPTURE QUERIES
# ============================================================

def get_latest_capture():
    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM capture_events
            ORDER BY id DESC
            LIMIT 1
            """,
            conn,
        )

        if df.empty:
            return None

        return df.iloc[0].to_dict()

    finally:
        conn.close()


def get_capture_by_id(capture_id):
    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM capture_events
            WHERE capture_id = ?
            LIMIT 1
            """,
            conn,
            params=(capture_id,),
        )

        if df.empty:
            return None

        return df.iloc[0].to_dict()

    finally:
        conn.close()


def get_capture_index():
    """
    Load a lightweight list of all captures for the new
    historical Capture Browser.
    """

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT
                capture_id,
                timestamp
            FROM capture_events
            ORDER BY timestamp DESC
            """,
            conn,
        )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

        return df

    finally:
        conn.close()


# ============================================================
# PLANT / HISTORY QUERIES
# ============================================================

def get_plant_samples(capture_id):
    conn = get_connection()

    try:
        return pd.read_sql_query(
            """
            SELECT
                ps.id,
                ps.capture_id,
                ps.plant_id,
                ps.crop_path,
                ps.model_image_path,
                ps.plant_active,

                ps.image_prediction,
                ps.image_confidence,
                ps.image_unhealthy_probability,

                ps.sensor_prediction,
                ps.sensor_confidence,
                ps.sensor_stress_probability,

                ps.decision_state,

                -- Legacy classical-fusion fields remain available for old captures.
                ps.fusion_prediction,
                ps.fusion_confidence,
                ps.final_status,

                ps.gradcam_path,
                pp.plant_name,
                pp.planted_date,
                pp.notes

            FROM plant_samples ps

            LEFT JOIN plant_positions pp
                ON ps.plant_id = pp.plant_id

            WHERE ps.capture_id = ?
            ORDER BY ps.plant_id
            """,
            conn,
            params=(capture_id,),
        )

    finally:
        conn.close()

def get_sensor_history():
    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT
                capture_id,
                timestamp,
                ph,
                ec_ms_cm,
                tds_ppm,
                water_temp_c,
                air_temp_c,
                humidity_pct,
                water_level,
                camera_lux

            FROM capture_events

            ORDER BY timestamp
            """,
            conn,
        )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

        return df

    finally:
        conn.close()


# ============================================================
# PLANT-SPECIFIC HISTORY
# ============================================================


def get_plant_history(plant_id):
    """Return the complete historical timeline for one plant position."""

    conn = get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT
                ce.timestamp,
                ce.capture_id,
                ce.ph,
                ce.ec_ms_cm,
                ce.tds_ppm,
                ce.water_temp_c,
                ce.air_temp_c,
                ce.humidity_pct,
                ce.water_level,
                ce.camera_lux,
                ce.environmental_warning,

                ps.plant_id,
                ps.plant_active,
                ps.crop_path,

                ps.image_prediction,
                ps.image_confidence,
                ps.image_unhealthy_probability,

                ps.sensor_prediction,
                ps.sensor_confidence,
                ps.sensor_stress_probability,

                ps.decision_state,

                ps.fusion_prediction,
                ps.fusion_confidence,
                ps.final_status,
                ps.gradcam_path

            FROM plant_samples ps

            JOIN capture_events ce
                ON ce.capture_id = ps.capture_id

            WHERE ps.plant_id = ?
            ORDER BY ce.timestamp
            """,
            conn,
            params=(plant_id,),
        )

        if not df.empty:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

        return df

    finally:
        conn.close()

def get_model_benchmark_results():
    """Load Raspberry Pi model benchmark results when available."""

    if not MODEL_BENCHMARK_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(MODEL_BENCHMARK_FILE)

    except Exception:
        return pd.DataFrame()


# ============================================================
# DISPLAY HELPERS
# ============================================================

def number(value, decimals=2):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):.{decimals}f}"

    except Exception:
        return str(value)


def confidence(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""

        return f"{float(value) * 100:.2f}%"

    except Exception:
        return ""


def in_range(value, minimum, maximum):
    try:
        value = float(value)
        return minimum <= value <= maximum

    except Exception:
        return False


def environmental_status(capture):
    water_level = str(
        capture.get("water_level", "")
    ).upper()

    return [
        (
            "pH",
            in_range(
                capture.get("ph"),
                PH_MIN,
                PH_MAX,
            ),
            number(capture.get("ph")),
        ),
        (
            "TDS",
            in_range(
                capture.get("tds_ppm"),
                TDS_MIN,
                TDS_MAX,
            ),
            f"{number(capture.get('tds_ppm'), 1)} ppm",
        ),
        (
            "Air Temperature",
            in_range(
                capture.get("air_temp_c"),
                AIR_TEMP_MIN,
                AIR_TEMP_MAX,
            ),
            f"{number(capture.get('air_temp_c'))} °C",
        ),
        (
            "Humidity",
            in_range(
                capture.get("humidity_pct"),
                HUMIDITY_MIN,
                HUMIDITY_MAX,
            ),
            f"{number(capture.get('humidity_pct'))}%",
        ),
        (
            "Water Temperature",
            in_range(
                capture.get("water_temp_c"),
                WATER_TEMP_MIN,
                WATER_TEMP_MAX,
            ),
            f"{number(capture.get('water_temp_c'))} °C",
        ),
        (
            "Water Level",
            water_level in {
                "OK",
                "NORMAL",
                "FULL",
                "HIGH",
            },
            water_level,
        ),
    ]


def summary_card(title, value, caption, card_class):
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sensor_card(title, value, reference):
    st.markdown(
        f"""
        <div class="sensor-card">
            <div class="sensor-title">{title}</div>
            <div class="sensor-value">{value}</div>
            <div class="sensor-reference">{reference}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_final_status(status):
    text = str(status or "").strip()
    upper = text.upper()

    if upper in {"HEALTHY", "HEALTHY / STABLE"}:
        st.success("✅ HEALTHY / STABLE")

    elif "HIGH RISK" in upper:
        st.error("🚨 HIGH RISK — VISIBLE + ENVIRONMENTAL STRESS")

    elif "VISIBLE PLANT STRESS" in upper:
        st.error("🍃 VISIBLE PLANT STRESS")
        st.caption("Current shared environmental conditions are normal.")

    elif "EARLY ENVIRONMENTAL WARNING" in upper:
        st.warning("🌡️ EARLY ENVIRONMENTAL WARNING")
        st.caption("Plant currently appears healthy, but environmental risk is elevated.")

    elif "ENVIRONMENTAL STRESS WARNING" in upper:
        st.warning("🌡️ ENVIRONMENTAL STRESS WARNING")
        if "IMAGE NOT EVALUATED" in upper:
            st.caption("Image AI was not available for this capture.")

    elif "ENVIRONMENT NORMAL" in upper and "IMAGE NOT EVALUATED" in upper:
        st.info("🌡️ ENVIRONMENT NORMAL — IMAGE NOT EVALUATED")

    elif "NO PLANT" in upper:
        st.info("— NO PLANT")

    elif "UNKNOWN" in upper:
        st.warning("❓ OCCUPANCY UNKNOWN")

    elif "SENSOR ONLY" in upper:
        st.warning("🌡️ SENSOR ONLY")

    elif "STRESS" in upper:
        # Legacy historical status.
        st.error("⚠️ LEGACY STRESS WARNING")

    else:
        st.info(text if text else "Not evaluated")

def get_image_quality_status(capture):
    capture_time = pd.to_datetime(
        capture["timestamp"]
    )

    time_ok = (
        IMAGE_START_HOUR
        <= capture_time.hour
        < IMAGE_END_HOUR
    )

    try:
        lux = float(
            capture.get("camera_lux")
        )

    except Exception:
        lux = None

    lux_ok = (
        lux is not None
        and lux >= MIN_IMAGE_LUX
    )

    image_ai_ready = (
        time_ok
        and lux_ok
    )

    return (
        time_ok,
        lux,
        lux_ok,
        image_ai_ready,
    )


def get_plant_counts(plants):
    if plants.empty:
        return 0, 0, 0, 0

    active_count = int((plants["plant_active"] == 1).sum())
    no_plant_count = int((plants["plant_active"] == 0).sum())

    def resolved_status(row):
        decision = row.get("decision_state")
        if decision is not None and not pd.isna(decision) and str(decision).strip():
            return str(decision).upper()
        return str(row.get("final_status") or "").upper()

    statuses = plants.apply(resolved_status, axis=1)

    healthy_count = int(
        statuses.isin({"HEALTHY", "HEALTHY / STABLE"}).sum()
    )

    warning_count = int(
        statuses.str.contains(
            "EARLY ENVIRONMENTAL WARNING|VISIBLE PLANT STRESS|HIGH RISK|ENVIRONMENTAL STRESS WARNING|STRESS WARNING",
            regex=True,
        ).sum()
    )

    return active_count, no_plant_count, healthy_count, warning_count

# ============================================================
# PLANT CARD + GRAD-CAM
# ============================================================

def plant_card(plant):
    plant_id = plant.get("plant_id")
    active = plant.get("plant_active")

    with st.container(border=True):
        st.markdown(f"### 🌱 {plant_id}")

        crop_path = plant.get("crop_path")
        if crop_path and Path(str(crop_path)).exists():
            st.image(str(crop_path), use_container_width=True)
        else:
            st.caption("Image unavailable")

        if pd.isna(active):
            show_final_status("OCCUPANCY UNKNOWN")
            return

        if int(active) == 0:
            st.caption("Plant position inactive")
            show_final_status("NO PLANT")
            return

        plant_name = plant.get("plant_name")
        planted_date = plant.get("planted_date")

        if plant_name and not pd.isna(plant_name):
            st.caption(f"Plant: {plant_name}")
        if planted_date and not pd.isna(planted_date):
            st.caption(f"Planted: {planted_date}")

        decision_state = plant.get("decision_state")
        display_status = (
            decision_state
            if decision_state is not None
            and not pd.isna(decision_state)
            and str(decision_state).strip()
            else plant.get("final_status")
        )

        show_final_status(display_status)
        st.divider()

        image_prediction = plant.get("image_prediction") or "Not evaluated"
        image_conf = confidence(plant.get("image_confidence"))
        image_risk = plant.get("image_unhealthy_probability")

        st.markdown("**📷 Image AI — Visible Condition**")
        st.write(
            f"{image_prediction} — {image_conf}"
            if image_conf else image_prediction
        )
        if image_risk is not None and not pd.isna(image_risk):
            st.caption(f"Visible unhealthy probability: {float(image_risk) * 100:.2f}%")

        sensor_prediction = plant.get("sensor_prediction") or "Not evaluated"
        sensor_conf = confidence(plant.get("sensor_confidence"))
        sensor_risk = plant.get("sensor_stress_probability")

        st.markdown("**🌡️ Sensor AI — Environmental Risk**")
        st.write(
            f"{sensor_prediction} — {sensor_conf}"
            if sensor_conf else sensor_prediction
        )
        if sensor_risk is not None and not pd.isna(sensor_risk):
            st.caption(f"Environmental stress probability: {float(sensor_risk) * 100:.2f}%")

        if (
            decision_state is not None
            and not pd.isna(decision_state)
            and str(decision_state).strip()
        ):
            st.markdown("**🧠 Complementary Decision**")
            st.write(str(decision_state))

        # ------------------------------------------------------------
        # LEGACY CLASSICAL FUSION - RESEARCH HISTORY
        # ------------------------------------------------------------
        #
        # Historical captures can contain BOTH:
        #   1. the newly backfilled complementary decision state, and
        #   2. the original classical 60/40 fusion result.
        #
        # Show the legacy result whenever it exists so the dashboard
        # preserves the research audit trail. It is explicitly labelled
        # historical and is NOT used for the final operational decision.
        # ------------------------------------------------------------

        legacy_fusion = plant.get("fusion_prediction")

        if (
            legacy_fusion is not None
            and not pd.isna(legacy_fusion)
            and str(legacy_fusion).strip()
        ):
            with st.expander(
                "Legacy classical fusion (historical)"
            ):
                legacy_conf = confidence(
                    plant.get("fusion_confidence")
                )

                st.write(
                    f"{legacy_fusion} — {legacy_conf}"
                    if legacy_conf
                    else str(legacy_fusion)
                )

                st.caption(
                    "Retired 60/40 probability-level fusion retained "
                    "for research history only. It does not control the "
                    "current complementary decision state."
                )

        gradcam_path = plant.get("gradcam_path")
        if (
            gradcam_path
            and not pd.isna(gradcam_path)
            and Path(str(gradcam_path)).exists()
        ):
            st.divider()
            with st.expander("🔍 View Image AI Explanation"):
                st.markdown("**Grad-CAM Image Explanation**")
                st.caption(
                    "Highlighted regions indicate areas that contributed strongly "
                    "to the final MobileNetV2 visible-health decision."
                )
                st.image(str(gradcam_path), use_container_width=True)
                st.caption(
                    "Red/yellow regions indicate stronger model attention and do "
                    "not by themselves prove diseased or stressed tissue."
                )

# ============================================================
# SHAP DISPLAY
# ============================================================

def render_shap_explanation(capture):
    st.write("")
    st.markdown(
        "## Explainable Sensor AI"
    )

    st.caption(
        "SHAP explains how each of the five environmental model features "
        "influenced the Random Forest environmental-risk prediction "
        "for this capture."
    )

    shap_path = capture.get(
        "shap_path"
    )

    if (
        shap_path
        and not pd.isna(shap_path)
        and Path(shap_path).exists()
    ):
        with st.container(border=True):

            shap_col1, shap_col2 = st.columns(
                [1.5, 1]
            )

            with shap_col1:
                st.image(
                    str(shap_path),
                    use_container_width=True,
                )

            with shap_col2:
                st.markdown(
                    "### 🌡️ Random Forest Explanation"
                )

                st.markdown(
                    """
                    **How to read the chart**

                    - **Positive SHAP value:** pushes the model
                      toward the displayed class.
                    - **Negative SHAP value:** pushes the model
                      away from the displayed class.
                    - **Larger absolute SHAP value:** stronger
                      influence on this prediction.
                    """
                )

                st.info(
                    "SHAP explains model behaviour. "
                    "It does not establish biological "
                    "cause and effect."
                )

    else:
        st.info(
            "SHAP explanation is not available "
            "for this capture."
        )


# ============================================================
# REUSABLE FULL CAPTURE VIEW
# ============================================================

def render_capture_overview(
    capture,
    plants,
    historical=False,
):
    """
    Render the same information for either the latest capture
    or a selected old capture.
    """

    (
        active_count,
        no_plant_count,
        healthy_count,
        warning_count,
    ) = get_plant_counts(plants)

    (
        time_ok,
        lux,
        lux_ok,
        image_ai_ready,
    ) = get_image_quality_status(capture)

    if historical:
        st.markdown(
            "## Selected Historical Capture"
        )

        st.caption(
            f"{capture['capture_id']} | "
            f"{capture['timestamp']}"
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        summary_card(
            "Active Plants",
            active_count,
            "Active positions in this capture",
            "metric-blue",
        )

    with col2:
        summary_card(
            "Healthy / Stable",
            healthy_count,
            "Final complementary decisions",
            "metric-green",
        )

    with col3:
        summary_card(
            "Attention States",
            warning_count,
            "Visible/environmental warnings",
            "metric-orange",
        )

    with col4:
        summary_card(
            "Inactive Positions",
            no_plant_count,
            "Positions without a plant",
            "metric-purple",
        )

    st.write("")

    st.markdown(
        "## Environmental Conditions"
    )

    st.caption(
        "Sensor values recorded at the time "
        "of this capture."
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        sensor_card(
            "pH",
            number(capture.get("ph")),
            f"Reference {PH_MIN:g} – {PH_MAX:g}",
        )

    with s2:
        sensor_card(
            "EC",
            f"{number(capture.get('ec_ms_cm'), 3)} mS/cm",
            "Nutrient conductivity",
        )

    with s3:
        sensor_card(
            "TDS",
            f"{number(capture.get('tds_ppm'), 1)} ppm",
            f"Reference {TDS_MIN:g} – {TDS_MAX:g} ppm",
        )

    with s4:
        sensor_card(
            "Water Level",
            capture.get(
                "water_level",
                "N/A",
            ),
            "Reservoir level status",
        )

    st.write("")

    s5, s6, s7, s8 = st.columns(4)

    with s5:
        sensor_card(
            "Water Temperature",
            f"{number(capture.get('water_temp_c'))} °C",
            f"Reference {WATER_TEMP_MIN:g} – {WATER_TEMP_MAX:g} °C",
        )

    with s6:
        sensor_card(
            "Air Temperature",
            f"{number(capture.get('air_temp_c'))} °C",
            f"Reference {AIR_TEMP_MIN:g} – {AIR_TEMP_MAX:g} °C",
        )

    with s7:
        sensor_card(
            "Humidity",
            f"{number(capture.get('humidity_pct'))}%",
            f"Reference {HUMIDITY_MIN:g} – {HUMIDITY_MAX:g}%",
        )

    with s8:
        sensor_card(
            "Camera Light",
            f"{number(lux)} Lux",
            f"Minimum image AI: {MIN_IMAGE_LUX}",
        )

    st.write("")
    st.write("")

    left, right = st.columns(
        [1.4, 1]
    )

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-title">
                    Environmental Status
                </div>

                <div class="panel-subtitle">
                    Rule-based assessment of the selected
                    capture's hydroponic conditions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            statuses = environmental_status(
                capture
            )

            issue_found = False

            for (
                name,
                normal,
                value,
            ) in statuses:

                if normal:
                    st.success(
                        f"✅ {name}: {value}"
                    )

                else:
                    issue_found = True

                    st.warning(
                        f"⚠️ {name}: {value}"
                    )

            if not issue_found:
                st.success(
                    "All monitored environmental "
                    "conditions are within configured ranges."
                )

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-title">
                    Edge-AI Status
                </div>

                <div class="panel-subtitle">
                    AI availability for this capture.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="system-row">
                    <span>Sensor Data</span>
                    <span class="ready-badge">AVAILABLE</span>
                </div>

                <div class="system-row">
                    <span>Random Forest Sensor AI</span>
                    <span class="ready-badge">AVAILABLE</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if image_ai_ready:
                st.markdown(
                    """
                    <div class="system-row">
                        <span>Image Quality Gate</span>
                        <span class="ready-badge">PASSED</span>
                    </div>

                    <div class="system-row">
                        <span>MobileNetV2 + Complementary Decision</span>
                        <span class="ready-badge">AVAILABLE</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    """
                    <div class="system-row">
                        <span>Image Quality Gate</span>
                        <span class="disabled-badge">NOT PASSED</span>
                    </div>

                    <div class="system-row">
                        <span>MobileNetV2 + Complementary Decision</span>
                        <span class="disabled-badge">GATED</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if not time_ok:
                    st.info(
                        "🌙 Capture is outside 06:00–16:59."
                    )

                elif not lux_ok:
                    st.warning(
                        f"☁️ Camera light was below "
                        f"{MIN_IMAGE_LUX} Lux."
                    )

    st.write("")

    st.markdown(
        "## Grow-System Image"
    )

    full_image = capture.get(
        "full_image_path"
    )

    if (
        full_image
        and Path(full_image).exists()
    ):
        with st.expander(
            "📷 View full camera capture",
            expanded=False,
        ):
            st.image(
                full_image,
                use_container_width=True,
            )

    else:
        st.warning(
            "Full camera image unavailable."
        )

    # Show capture-level SHAP for both latest and old captures.
    render_shap_explanation(
        capture
    )

    st.write("")

    st.markdown(
        "## P01–P12 Plant Results"
    )

    st.caption(
        "Historical occupancy, visible-health AI, environmental-risk AI, "
        "complementary decision state and Grad-CAM for this capture."
    )

    if plants.empty:
        st.info(
            "No plant samples exist for this capture."
        )
        return

    plant_records = (
        plants
        .sort_values("plant_id")
        .to_dict(
            orient="records"
        )
    )

    for start in range(
        0,
        len(plant_records),
        4,
    ):
        columns = st.columns(4)

        row = plant_records[
            start:
            start + 4
        ]

        for (
            column,
            plant,
        ) in zip(
            columns,
            row,
        ):
            with column:
                plant_card(
                    plant
                )


# ============================================================
# LOAD LATEST DATA
# ============================================================

check_database()
ensure_final_decision_schema()

# Create shared settings and the independent ground-truth
# labelling table on first launch.
ensure_system_settings()
ensure_manual_labels_table()

SETTINGS = load_system_settings()

# Update the global display/inference-rule values used by the existing
# helper functions. run_inference.py reads the same SQLite settings.
PH_MIN = SETTINGS["ph_min"]
PH_MAX = SETTINGS["ph_max"]
TDS_MIN = SETTINGS["tds_min"]
TDS_MAX = SETTINGS["tds_max"]
AIR_TEMP_MIN = SETTINGS["air_temp_min"]
AIR_TEMP_MAX = SETTINGS["air_temp_max"]
HUMIDITY_MIN = SETTINGS["humidity_min"]
HUMIDITY_MAX = SETTINGS["humidity_max"]
WATER_TEMP_MIN = SETTINGS["water_temp_min"]
WATER_TEMP_MAX = SETTINGS["water_temp_max"]
MIN_IMAGE_LUX = SETTINGS["min_image_lux"]
IMAGE_START_HOUR = SETTINGS["image_start_hour"]
IMAGE_END_HOUR = SETTINGS["image_end_hour"]

latest = get_latest_capture()

if latest is None:
    st.error(
        "No capture data available."
    )
    st.stop()

latest_capture_id = latest[
    "capture_id"
]

latest_plants = get_plant_samples(
    latest_capture_id
)


# ============================================================
# SIDEBAR + NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-brand-title">
        🌱 Hydroponic Edge-AI
    </div>

    <div class="sidebar-brand-subtitle">
        Intelligent Lettuce Monitoring
    </div>

    <div class="online-box">
        ● EDGE SYSTEM ONLINE
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Plant Health",
        "Plant History",
        "Capture Browser",
        "History",
        "Model Comparison",
        "Evaluation Labelling",
        "Settings",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.caption(
    "LATEST CAPTURE"
)

st.sidebar.code(
    latest_capture_id
)

st.sidebar.write(
    f"**{latest['timestamp']}**"
)

if st.sidebar.button(
    "🔄 Refresh",
    use_container_width=True,
):
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    "IMAGE AI QUALITY RULE"
)

st.sidebar.write(
    f"🕕 {IMAGE_START_HOUR:02d}:00 – {IMAGE_END_HOUR - 1:02d}:59"
)

st.sidebar.write(
    f"☀️ Lux ≥ {MIN_IMAGE_LUX}"
)

st.sidebar.caption(
    "Sensor AI operates independently "
    "of camera lighting."
)

st.sidebar.caption(
    f"Image unhealthy threshold: {SETTINGS['image_unhealthy_threshold']:.2f} | "
    f"Sensor stress threshold: {SETTINGS['sensor_stress_threshold']:.2f}"
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    (
        f'<div class="hero-card">'
        f'<div class="hero-title">Hydroponic Edge-AI Monitoring</div>'
        f'<div class="hero-subtitle">'
        f'Real-time and historical multimodal lettuce health monitoring using IoT sensors, '
        f'computer vision and edge AI.'
        f'</div>'
        f'<div class="hero-badge">● Latest capture: {latest["timestamp"]}</div>'
        f'</div>'
    ),
    unsafe_allow_html=True,
)


# ============================================================
# PAGE: DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        "## Latest Capture"
    )

    st.caption(
        "This page always displays the newest capture."
    )

    render_capture_overview(
        latest,
        latest_plants,
        historical=False,
    )


# ============================================================
# PAGE: PLANT HEALTH
# ============================================================

elif page == "Plant Health":

    st.markdown(
        "## Latest P01–P12 Plant Health"
    )

    st.caption(
        "Individual plant results from the latest capture."
    )

    if latest_plants.empty:
        st.info(
            "No plant data available."
        )

    else:
        plant_records = (
            latest_plants
            .sort_values("plant_id")
            .to_dict(
                orient="records"
            )
        )

        for start in range(
            0,
            len(plant_records),
            4,
        ):
            columns = st.columns(4)

            row = plant_records[
                start:
                start + 4
            ]

            for (
                column,
                plant,
            ) in zip(
                columns,
                row,
            ):
                with column:
                    plant_card(
                        plant
                    )


# ============================================================
# PAGE: PLANT HISTORY
# ============================================================

elif page == "Plant History":

    st.markdown(
        "## 🌿 Plant-Specific Health History"
    )

    st.caption(
        "Track one plant position across time and compare visible-health AI, "
        "environmental-risk AI and the complementary decision state."
    )

    plant_col1, plant_col2 = st.columns([1, 1.2])

    with plant_col1:
        selected_plant = st.selectbox(
            "Plant position",
            [f"P{i:02d}" for i in range(1, 13)],
        )

    with plant_col2:
        plant_period = st.selectbox(
            "Display period",
            [
                "Last 24 Hours",
                "Last 7 Days",
                "All Data",
            ],
        )

    plant_history = get_plant_history(
        selected_plant
    )

    if plant_history.empty:
        st.info(
            f"No historical records found for {selected_plant}."
        )

    else:
        filtered_plant = plant_history.copy()
        now = pd.Timestamp.now()

        if plant_period == "Last 24 Hours":
            filtered_plant = filtered_plant[
                filtered_plant["timestamp"]
                >= now - pd.Timedelta(hours=24)
            ]

        elif plant_period == "Last 7 Days":
            filtered_plant = filtered_plant[
                filtered_plant["timestamp"]
                >= now - pd.Timedelta(days=7)
            ]

        if filtered_plant.empty:
            st.info(
                "No records exist in the selected time period."
            )

        else:
            active_rows = filtered_plant[
                filtered_plant["plant_active"] == 1
            ]

            decision_rows = active_rows[
                active_rows["decision_state"].notna()
            ]

            stress_rows = active_rows[
                active_rows["final_status"]
                .fillna("")
                .str.upper()
                .str.contains("STRESS")
            ]

            p1, p2, p3, p4 = st.columns(4)

            with p1:
                st.metric(
                    "Observations",
                    len(filtered_plant),
                )

            with p2:
                st.metric(
                    "Active Observations",
                    len(active_rows),
                )

            with p3:
                st.metric(
                    "Decision Evaluations",
                    len(decision_rows),
                )

            with p4:
                st.metric(
                    "Attention States",
                    len(stress_rows),
                )

            st.write("")
            st.markdown(
                "### AI Confidence Over Time"
            )

            confidence_df = filtered_plant[
                [
                    "timestamp",
                    "image_confidence",
                    "sensor_confidence",
                ]
            ].copy()

            confidence_df["Image AI %"] = (
                confidence_df["image_confidence"] * 100
            )
            confidence_df["Sensor AI %"] = (
                confidence_df["sensor_confidence"] * 100
            )

            confidence_df = confidence_df.set_index("timestamp")

            st.line_chart(
                confidence_df[["Image AI %", "Sensor AI %"]]
            )

            st.caption(
                "Image AI confidence describes the visible-health decision; "
                "Sensor AI confidence describes the shared environmental state. "
                "They are not probabilities of the same biological target."
            )

            risk_df = filtered_plant[
                [
                    "timestamp",
                    "image_unhealthy_probability",
                    "sensor_stress_probability",
                ]
            ].copy()

            if risk_df[[
                "image_unhealthy_probability",
                "sensor_stress_probability",
            ]].notna().any().any():
                st.markdown("### Risk Scores Over Time")
                risk_df["Visible Unhealthy Risk %"] = (
                    risk_df["image_unhealthy_probability"] * 100
                )
                risk_df["Environmental Stress Risk %"] = (
                    risk_df["sensor_stress_probability"] * 100
                )
                risk_df = risk_df.set_index("timestamp")
                st.line_chart(
                    risk_df[[
                        "Visible Unhealthy Risk %",
                        "Environmental Stress Risk %",
                    ]]
                )

            st.markdown(
                "### Observation Timeline"
            )

            timeline = filtered_plant[
                [
                    "timestamp",
                    "capture_id",
                    "plant_active",
                    "image_prediction",
                    "image_confidence",
                    "image_unhealthy_probability",
                    "sensor_prediction",
                    "sensor_confidence",
                    "sensor_stress_probability",
                    "decision_state",
                    "final_status",
                    "camera_lux",
                ]
            ].copy()

            for col in [
                "image_confidence",
                "sensor_confidence",
                "image_unhealthy_probability",
                "sensor_stress_probability",
            ]:
                timeline[col] = (
                    timeline[col] * 100
                ).round(2)

            timeline = timeline.rename(
                columns={
                    "timestamp": "Time",
                    "capture_id": "Capture ID",
                    "plant_active": "Active",
                    "image_prediction": "Image AI",
                    "image_confidence": "Image Conf %",
                    "image_unhealthy_probability": "Image Unhealthy Risk %",
                    "sensor_prediction": "Environmental AI",
                    "sensor_confidence": "Sensor Conf %",
                    "sensor_stress_probability": "Sensor Stress Risk %",
                    "decision_state": "Complementary Decision",
                    "final_status": "Operational Status",
                    "camera_lux": "Lux",
                }
            )

            st.dataframe(
                timeline.sort_values(
                    "Time",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Inspect One Plant Observation"
            )

            observation_options = {}

            for _, row in filtered_plant.sort_values(
                "timestamp",
                ascending=False,
            ).iterrows():
                label = (
                    f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"{row['capture_id']}"
                )
                observation_options[label] = row

            selected_observation_label = st.selectbox(
                "Historical observation",
                list(observation_options.keys()),
            )

            selected_observation = observation_options[
                selected_observation_label
            ]

            obs_left, obs_right = st.columns([1, 1])

            with obs_left:
                crop_path = selected_observation.get(
                    "crop_path"
                )

                if (
                    crop_path
                    and not pd.isna(crop_path)
                    and Path(crop_path).exists()
                ):
                    st.image(
                        str(crop_path),
                        caption="Historical plant crop",
                        use_container_width=True,
                    )

                else:
                    st.info(
                        "Historical crop image is unavailable."
                    )

            with obs_right:
                show_final_status(
                    selected_observation.get(
                        "final_status"
                    )
                )

                st.write(
                    "**Image AI:**",
                    selected_observation.get(
                        "image_prediction"
                    ),
                )

                st.write(
                    "**Sensor AI:**",
                    selected_observation.get(
                        "sensor_prediction"
                    ),
                )

                decision = selected_observation.get(
                    "decision_state"
                )

                if (
                    decision
                    and not pd.isna(decision)
                ):
                    st.write(
                        "**Complementary Decision:**",
                        decision,
                    )
                else:
                    st.write(
                        "**Legacy Fusion AI:**",
                        selected_observation.get(
                            "fusion_prediction"
                        ),
                    )

                gradcam_path = selected_observation.get(
                    "gradcam_path"
                )

                if (
                    gradcam_path
                    and not pd.isna(gradcam_path)
                    and Path(gradcam_path).exists()
                ):
                    with st.expander(
                        "🔍 View Grad-CAM"
                    ):
                        st.image(
                            str(gradcam_path),
                            use_container_width=True,
                        )


# ============================================================
# PAGE: CAPTURE BROWSER
# ============================================================

elif page == "Capture Browser":

    st.markdown(
        "## 🗂️ Historical Capture Browser"
    )

    st.caption(
        "Select an old collection event and inspect the "
        "same sensor, image, AI and explainability data "
        "available for the latest capture."
    )

    capture_index = get_capture_index()

    if capture_index.empty:
        st.info(
            "No historical captures are available."
        )

    else:
        capture_index = capture_index.dropna(
            subset=["timestamp"]
        ).copy()

        capture_index[
            "capture_date"
        ] = (
            capture_index[
                "timestamp"
            ].dt.date
        )

        available_dates = sorted(
            capture_index[
                "capture_date"
            ].unique(),
            reverse=True,
        )

        latest_date = available_dates[0]
        oldest_date = available_dates[-1]

        info1, info2, info3 = st.columns(3)

        with info1:
            st.metric(
                "Total Captures",
                len(capture_index),
            )

        with info2:
            st.metric(
                "Oldest Date",
                str(oldest_date),
            )

        with info3:
            st.metric(
                "Latest Date",
                str(latest_date),
            )

        st.write("")

        filter_col1, filter_col2 = st.columns(
            [1.2, 1]
        )

        with filter_col1:
            selected_date = st.date_input(
                "Capture date",
                value=latest_date,
                min_value=oldest_date,
                max_value=latest_date,
                help=(
                    "Choose the date containing "
                    "the capture you want to inspect."
                ),
            )

        with filter_col2:
            order_option = st.radio(
                "Capture order",
                [
                    "Newest first",
                    "Oldest first",
                ],
                horizontal=True,
            )

        daily_captures = (
            capture_index[
                capture_index[
                    "capture_date"
                ] == selected_date
            ]
            .copy()
        )

        if order_option == "Newest first":
            daily_captures = (
                daily_captures
                .sort_values(
                    "timestamp",
                    ascending=False,
                )
            )

        else:
            daily_captures = (
                daily_captures
                .sort_values(
                    "timestamp",
                    ascending=True,
                )
            )

        if daily_captures.empty:
            st.warning(
                "No captures found for "
                f"{selected_date}."
            )

        else:
            capture_options = []
            capture_label_to_id = {}

            for _, row in daily_captures.iterrows():
                timestamp = row[
                    "timestamp"
                ]

                capture_id = row[
                    "capture_id"
                ]

                label = (
                    f"{timestamp.strftime('%H:%M:%S')} | "
                    f"{capture_id}"
                )

                capture_options.append(
                    label
                )

                capture_label_to_id[
                    label
                ] = capture_id

            selected_label = st.selectbox(
                "Select capture",
                capture_options,
            )

            selected_capture_id = (
                capture_label_to_id[
                    selected_label
                ]
            )

            with st.expander(
                "🔎 Search directly by Capture ID"
            ):
                direct_capture_id = st.text_input(
                    "Capture ID",
                    placeholder=(
                        "CAP_20260901_143003"
                    ),
                ).strip()

                if direct_capture_id:
                    matching = capture_index[
                        capture_index[
                            "capture_id"
                        ].str.upper()
                        == direct_capture_id.upper()
                    ]

                    if matching.empty:
                        st.warning(
                            "Capture ID not found."
                        )

                    else:
                        selected_capture_id = (
                            matching.iloc[0][
                                "capture_id"
                            ]
                        )

                        st.success(
                            f"Capture found: "
                            f"{selected_capture_id}"
                        )

            st.divider()

            selected_capture = get_capture_by_id(
                selected_capture_id
            )

            if selected_capture is None:
                st.error(
                    "Unable to load selected capture."
                )

            else:
                selected_plants = get_plant_samples(
                    selected_capture_id
                )

                render_capture_overview(
                    selected_capture,
                    selected_plants,
                    historical=True,
                )


# ============================================================
# PAGE: MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.markdown(
        "## ⚙️ Image Model Edge Benchmark"
    )

    st.caption(
        "Compare MobileNetV2, ResNet18 and EfficientNet-B0 on the "
        "Raspberry Pi using real collected plant crops."
    )

    benchmark = get_model_benchmark_results()

    if benchmark.empty:
        st.info(
            "No benchmark CSV is available yet. Run the model comparison "
            "script on the Raspberry Pi, then refresh this page."
        )

        st.code(
            "/hydroponic_project/venv/bin/python "
            "/hydroponic_project/evaluation/compare_image_models.py "
            "--samples 50",
            language="bash",
        )

    else:
        numeric_cols = [
            "file_size_mb",
            "parameters_m",
            "load_ms",
            "mean_inference_ms",
            "median_inference_ms",
            "p95_inference_ms",
            "throughput_img_s",
        ]

        for col in numeric_cols:
            if col in benchmark.columns:
                benchmark[col] = pd.to_numeric(
                    benchmark[col],
                    errors="coerce",
                )

        valid_latency = benchmark.dropna(
            subset=["mean_inference_ms"]
        )

        valid_size = benchmark.dropna(
            subset=["file_size_mb"]
        )

        fastest_model = (
            valid_latency.loc[
                valid_latency["mean_inference_ms"].idxmin(),
                "model",
            ]
            if not valid_latency.empty
            else "N/A"
        )

        smallest_model = (
            valid_size.loc[
                valid_size["file_size_mb"].idxmin(),
                "model",
            ]
            if not valid_size.empty
            else "N/A"
        )

        b1, b2, b3 = st.columns(3)

        with b1:
            st.metric(
                "Fastest Model",
                fastest_model,
            )

        with b2:
            st.metric(
                "Smallest Model",
                smallest_model,
            )

        with b3:
            sample_count = (
                int(benchmark["samples"].max())
                if "samples" in benchmark.columns
                else "N/A"
            )
            st.metric(
                "Benchmark Images",
                sample_count,
            )

        st.markdown(
            "### Raspberry Pi Benchmark Results"
        )

        st.dataframe(
            benchmark,
            use_container_width=True,
            hide_index=True,
        )

        if not valid_latency.empty:
            st.markdown(
                "### Mean Model Inference Latency"
            )

            latency_chart = (
                valid_latency[
                    [
                        "model",
                        "mean_inference_ms",
                    ]
                ]
                .set_index("model")
            )

            st.bar_chart(
                latency_chart
            )

        if not valid_size.empty:
            st.markdown(
                "### Model File Size"
            )

            size_chart = (
                valid_size[
                    [
                        "model",
                        "file_size_mb",
                    ]
                ]
                .set_index("model")
            )

            st.bar_chart(
                size_chart
            )

        st.info(
            "Edge-model selection should consider BOTH deployment cost "
            "(latency, size, memory) and held-out classification quality. "
            "The fastest model is not automatically the best research model."
        )



# ============================================================
# PAGE: EVALUATION LABELLING
# ============================================================
#
# This page is intentionally BLINDED before label submission:
#
#       plant image
#            ↓
#       human judgement
#            ↓
#       save label
#            ↓
#       reveal AI predictions
#
# This reduces anchoring bias from seeing the model answer
# before creating the independent reference label.
# ============================================================

elif page == "Evaluation Labelling":

    st.markdown(
        "## 🧪 Ground-Truth Evaluation Labelling"
    )

    st.caption(
        "Create independent Healthy / Unhealthy / Uncertain "
        "labels for historical plant observations. AI predictions "
        "are hidden until a label already exists for the selected "
        "sample."
    )

    st.warning(
        "Research rule: judge the plant first. "
        "Do not use the Image, Sensor or Fusion predictions to "
        "decide the ground-truth label."
    )

    candidates = get_evaluation_candidates()

    if candidates.empty:

        st.info(
            "No eligible multimodal observations are available."
        )

    else:

        # ----------------------------------------------------
        # SAMPLING STRATEGY
        # ----------------------------------------------------

        sampling_strategy = st.radio(
            "Evaluation sampling strategy",
            [
                "Hourly representative (recommended)",
                "All valid multimodal observations",
            ],
            horizontal=True,
            help=(
                "Hourly representative reduces the number of "
                "near-identical 15-minute captures treated as "
                "separate evaluation examples."
            ),
        )

        if sampling_strategy.startswith(
            "Hourly"
        ):
            evaluation_pool = (
                apply_hourly_representative_sampling(
                    candidates
                )
            )

        else:
            evaluation_pool = candidates.copy()

        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        valid_timestamps = (
            evaluation_pool[
                "timestamp"
            ]
            .dropna()
        )

        available_date_strings = sorted(
            valid_timestamps
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist(),
            reverse=True,
        )

        filter1, filter2, filter3 = st.columns(
            [
                1.2,
                1,
                1,
            ]
        )

        with filter1:

            date_choice = st.selectbox(
                "Date",
                [
                    "All dates",
                    *available_date_strings,
                ],
            )

        with filter2:

            plant_options = sorted(
                evaluation_pool[
                    "plant_id"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            plant_choice = st.selectbox(
                "Plant",
                [
                    "All plants",
                    *plant_options,
                ],
            )

        with filter3:

            label_status = st.selectbox(
                "Label status",
                [
                    "Unlabelled only",
                    "All samples",
                    "Labelled only",
                ],
            )

        filtered_pool = (
            evaluation_pool.copy()
        )

        if date_choice != "All dates":

            filtered_pool = filtered_pool[
                filtered_pool[
                    "timestamp"
                ]
                .dt.strftime(
                    "%Y-%m-%d"
                )
                == date_choice
            ]

        if plant_choice != "All plants":

            filtered_pool = filtered_pool[
                filtered_pool[
                    "plant_id"
                ]
                == plant_choice
            ]

        if label_status == "Unlabelled only":

            filtered_pool = filtered_pool[
                filtered_pool[
                    "ground_truth_label"
                ].isna()
            ]

        elif label_status == "Labelled only":

            filtered_pool = filtered_pool[
                filtered_pool[
                    "ground_truth_label"
                ].notna()
            ]

        filtered_pool = filtered_pool.sort_values(
            [
                "timestamp",
                "plant_id",
            ]
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # LABELLING PROGRESS
        # ----------------------------------------------------

        total_pool = len(
            evaluation_pool
        )

        labelled_pool = int(
            evaluation_pool[
                "ground_truth_label"
            ].notna().sum()
        )

        healthy_pool = int(
            evaluation_pool[
                "ground_truth_label"
            ]
            .fillna("")
            .eq("healthy")
            .sum()
        )

        unhealthy_pool = int(
            evaluation_pool[
                "ground_truth_label"
            ]
            .fillna("")
            .eq("unhealthy")
            .sum()
        )

        uncertain_pool = int(
            evaluation_pool[
                "ground_truth_label"
            ]
            .fillna("")
            .eq("uncertain")
            .sum()
        )

        progress1, progress2, progress3, progress4 = (
            st.columns(4)
        )

        with progress1:
            st.metric(
                "Evaluation Samples",
                total_pool,
            )

        with progress2:
            st.metric(
                "Labelled",
                labelled_pool,
            )

        with progress3:
            st.metric(
                "Healthy / Unhealthy",
                (
                    f"{healthy_pool} / "
                    f"{unhealthy_pool}"
                ),
            )

        with progress4:
            st.metric(
                "Uncertain",
                uncertain_pool,
            )

        if total_pool > 0:

            st.progress(
                min(
                    labelled_pool
                    / total_pool,
                    1.0,
                ),
                text=(
                    f"{labelled_pool} of "
                    f"{total_pool} evaluation samples labelled"
                ),
            )

        st.write("")

        # ----------------------------------------------------
        # SELECT ONE OBSERVATION
        # ----------------------------------------------------

        if filtered_pool.empty:

            st.success(
                "No samples match the current filters. "
                "If 'Unlabelled only' is selected, this may mean "
                "you completed this filtered evaluation set."
            )

        else:

            option_labels = []

            option_map = {}

            for _, item in filtered_pool.iterrows():

                timestamp = item[
                    "timestamp"
                ]

                label_text = (
                    f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| {item['plant_id']} "
                    f"| {item['capture_id']}"
                )

                option_labels.append(
                    label_text
                )

                option_map[
                    label_text
                ] = (
                    item[
                        "capture_id"
                    ],
                    item[
                        "plant_id"
                    ],
                )

            selected_option = st.selectbox(
                "Observation to review",
                option_labels,
            )

            (
                selected_capture_id,
                selected_plant_id,
            ) = option_map[
                selected_option
            ]

            selected_rows = filtered_pool[
                (
                    filtered_pool[
                        "capture_id"
                    ]
                    == selected_capture_id
                )
                & (
                    filtered_pool[
                        "plant_id"
                    ]
                    == selected_plant_id
                )
            ]

            selected = (
                selected_rows.iloc[0]
            )

            existing_label = (
                selected.get(
                    "ground_truth_label"
                )
            )

            has_existing_label = (
                existing_label is not None
                and not pd.isna(
                    existing_label
                )
                and str(
                    existing_label
                ).strip()
                != ""
            )

            # ------------------------------------------------
            # BLINDED REVIEW IMAGE
            # ------------------------------------------------

            review_col, form_col = st.columns(
                [
                    1.35,
                    1,
                ]
            )

            with review_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### Review Image"
                    )

                    st.caption(
                        f"Capture: "
                        f"{selected_capture_id}"
                    )

                    st.caption(
                        f"Plant: "
                        f"{selected_plant_id}"
                    )

                    st.caption(
                        f"Captured: "
                        f"{selected['timestamp']}"
                    )

                    crop_path = (
                        selected.get(
                            "crop_path"
                        )
                    )

                    model_image_path = (
                        selected.get(
                            "model_image_path"
                        )
                    )

                    display_image = None

                    if (
                        crop_path
                        and not pd.isna(
                            crop_path
                        )
                        and Path(
                            str(
                                crop_path
                            )
                        ).exists()
                    ):

                        display_image = str(
                            crop_path
                        )

                    elif (
                        model_image_path
                        and not pd.isna(
                            model_image_path
                        )
                        and Path(
                            str(
                                model_image_path
                            )
                        ).exists()
                    ):

                        display_image = str(
                            model_image_path
                        )

                    if display_image:

                        st.image(
                            display_image,
                            use_container_width=True,
                        )

                    else:

                        st.error(
                            "Plant image is unavailable "
                            "for this observation."
                        )

                    full_image_path = (
                        selected.get(
                            "full_image_path"
                        )
                    )

                    if (
                        full_image_path
                        and not pd.isna(
                            full_image_path
                        )
                        and Path(
                            str(
                                full_image_path
                            )
                        ).exists()
                    ):

                        with st.expander(
                            "View full grow-system image"
                        ):

                            st.image(
                                str(
                                    full_image_path
                                ),
                                use_container_width=True,
                            )

            # ------------------------------------------------
            # LABEL FORM
            # ------------------------------------------------

            with form_col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        "### Independent Ground Truth"
                    )

                    st.caption(
                        "Assign the label from visual/physical "
                        "evidence. AI predictions remain hidden "
                        "until this sample has been labelled."
                    )

                    label_options = [
                        "Select label...",
                        "healthy",
                        "unhealthy",
                        "uncertain",
                    ]

                    if has_existing_label:

                        existing_text = str(
                            existing_label
                        ).strip().lower()

                        try:
                            label_index = (
                                label_options.index(
                                    existing_text
                                )
                            )

                        except ValueError:
                            label_index = 0

                    else:
                        label_index = 0

                    selected_truth = st.selectbox(
                        "Ground-truth label",
                        label_options,
                        index=label_index,
                        key=(
                            "eval_truth_"
                            f"{selected_capture_id}_"
                            f"{selected_plant_id}"
                        ),
                    )

                    validation_options = [
                        "manual_visual",
                        "physical_observation",
                        "expert_review",
                        "expert_plus_visual",
                    ]

                    existing_method = (
                        selected.get(
                            "validation_method"
                        )
                    )

                    if (
                        has_existing_label
                        and existing_method
                        and not pd.isna(
                            existing_method
                        )
                        and str(
                            existing_method
                        ) in validation_options
                    ):

                        validation_index = (
                            validation_options.index(
                                str(
                                    existing_method
                                )
                            )
                        )

                    else:
                        validation_index = 0

                    validation_method = st.selectbox(
                        "Validation method",
                        validation_options,
                        index=validation_index,
                        help=(
                            "Use expert_review only when an "
                            "appropriate domain expert actually "
                            "validated this observation."
                        ),
                    )

                    existing_reviewer = (
                        selected.get(
                            "reviewer_role"
                        )
                    )

                    reviewer_role = st.text_input(
                        "Reviewer role (optional)",
                        value=(
                            ""
                            if (
                                existing_reviewer is None
                                or pd.isna(
                                    existing_reviewer
                                )
                            )
                            else str(
                                existing_reviewer
                            )
                        ),
                        placeholder=(
                            "Example: researcher / "
                            "hydroponic expert"
                        ),
                    )

                    existing_notes = (
                        selected.get(
                            "notes"
                        )
                    )

                    notes = st.text_area(
                        "Observation notes",
                        value=(
                            ""
                            if (
                                existing_notes is None
                                or pd.isna(
                                    existing_notes
                                )
                            )
                            else str(
                                existing_notes
                            )
                        ),
                        placeholder=(
                            "Example: slight leaf-margin "
                            "yellowing; otherwise upright."
                        ),
                        height=110,
                    )

                    save_clicked = st.button(
                        (
                            "💾 Update Label"
                            if has_existing_label
                            else "💾 Save Label"
                        ),
                        type="primary",
                        use_container_width=True,
                    )

                    if save_clicked:

                        if selected_truth == "Select label...":

                            st.error(
                                "Select Healthy, Unhealthy "
                                "or Uncertain before saving."
                            )

                        else:

                            save_manual_label(
                                selected_capture_id,
                                selected_plant_id,
                                selected_truth,
                                validation_method,
                                reviewer_role,
                                notes,
                            )

                            st.success(
                                "Ground-truth label saved. "
                                "The AI results can now be "
                                "revealed for comparison."
                            )

                            st.rerun()

            # ------------------------------------------------
            # REVEAL AI ONLY AFTER LABEL EXISTS
            # ------------------------------------------------

            if has_existing_label:

                st.write("")

                st.success(
                    f"Ground truth: "
                    f"{str(existing_label).upper()}"
                )

                render_post_label_ai_comparison(
                    selected
                )

            else:

                st.info(
                    "AI predictions are intentionally hidden "
                    "for this unlabelled observation."
                )

        # ----------------------------------------------------
        # LABEL EXPORT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "### Evaluation Label Export"
        )

        st.caption(
            "Export saved labels together with the stored model "
            "predictions and environmental values for later "
            "statistical evaluation."
        )

        export_df = get_manual_labels_export()

        if export_df.empty:

            st.info(
                "No ground-truth labels have been saved yet."
            )

        else:

            st.download_button(
                "⬇️ Download labelled evaluation CSV",
                data=export_df.to_csv(
                    index=False
                ).encode(
                    "utf-8"
                ),
                file_name=(
                    "hydroponic_ground_truth_evaluation.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander(
                "Preview saved labels"
            ):

                st.dataframe(
                    export_df[
                        [
                            "timestamp",
                            "capture_id",
                            "plant_id",
                            "ground_truth_label",
                            "validation_method",
                            "labeled_at",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# PAGE: SETTINGS
# ============================================================

elif page == "Settings":

    st.markdown(
        "## 🛠️ System Settings"
    )

    st.caption(
        "Manage current plant occupancy and operational AI thresholds. "
        "Changes apply to future captures/inference; historical results "
        "remain unchanged unless you explicitly reprocess them."
    )

    # --------------------------------------------------------
    # PLANT POSITION SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "### 🌱 Plant Position Management"
    )

    st.info(
        "Turning a position OFF means future collections will store "
        "plant_active = 0 for that position and image/fusion inference "
        "will be skipped. Existing historical captures are not changed."
    )

    positions = get_plant_positions()

    with st.form(
        "plant_position_form"
    ):
        activity_map = {}

        position_records = positions.to_dict(
            orient="records"
        )

        for start in range(
            0,
            len(position_records),
            4,
        ):
            cols = st.columns(4)

            for col, plant in zip(
                cols,
                position_records[
                    start:start + 4
                ],
            ):
                with col:
                    plant_id = plant["plant_id"]

                    current_active = bool(
                        plant["active"]
                    )

                    activity_map[plant_id] = st.checkbox(
                        f"{plant_id} Active",
                        value=current_active,
                        key=f"active_{plant_id}",
                    )

                    if (
                        plant.get("plant_name")
                        and not pd.isna(plant.get("plant_name"))
                    ):
                        st.caption(
                            f"Plant: {plant['plant_name']}"
                        )

        save_plants = st.form_submit_button(
            "💾 Save Plant Positions",
            use_container_width=True,
        )

    if save_plants:
        update_plant_positions(
            activity_map
        )

        st.success(
            "Plant positions updated. The next collection will use "
            "the new active/inactive configuration."
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # FINAL IMAGE + SENSOR DECISION SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 Final AI Decision Settings"
    )

    st.warning(
        "These values control future inference. The final deployment uses "
        "MobileNetV2 threshold 0.42 and sensor RF threshold 0.48. Classical "
        "60/40 probability fusion is retained only as a research result and "
        "is not used for the operational decision state."
    )

    with st.form(
        "ai_settings_form"
    ):
        ai1, ai2 = st.columns(2)

        with ai1:
            new_min_lux = st.number_input(
                "Minimum image Lux",
                min_value=0.0,
                max_value=100000.0,
                value=float(SETTINGS["min_image_lux"]),
                step=1.0,
            )

            new_start_hour = st.number_input(
                "Image AI start hour",
                min_value=0,
                max_value=23,
                value=int(SETTINGS["image_start_hour"]),
                step=1,
            )

            new_end_hour = st.number_input(
                "Image AI end hour (exclusive)",
                min_value=1,
                max_value=24,
                value=int(SETTINGS["image_end_hour"]),
                step=1,
            )

        with ai2:
            new_image_threshold = st.slider(
                "MobileNetV2 unhealthy threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(SETTINGS["image_unhealthy_threshold"]),
                step=0.01,
                help="Exploratory live-label analysis selected 0.42.",
            )

            new_sensor_threshold = st.slider(
                "Sensor environmental-stress threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(SETTINGS["sensor_stress_threshold"]),
                step=0.01,
                help="Chronological sensor validation selected 0.48.",
            )

            st.caption(
                "The two probabilities represent different targets and are "
                "therefore not averaged in the final production decision."
            )

        save_ai_settings = st.form_submit_button(
            "💾 Save Final AI Settings",
            use_container_width=True,
        )

    if save_ai_settings:
        if new_end_hour <= new_start_hour:
            st.error(
                "End hour must be later than start hour."
            )

        else:
            save_system_settings(
                {
                    "min_image_lux": new_min_lux,
                    "image_start_hour": int(new_start_hour),
                    "image_end_hour": int(new_end_hour),
                    "image_unhealthy_threshold": new_image_threshold,
                    "sensor_stress_threshold": new_sensor_threshold,
                }
            )

            st.success(
                "Final AI settings saved. run_inference.py will read these "
                "values on the next pipeline execution."
            )

            st.rerun()

    st.divider()

    # --------------------------------------------------------
    # ENVIRONMENTAL DISPLAY / ALERT SETTINGS
    # --------------------------------------------------------

    st.markdown(
        "### 🌡️ Environmental Reference Ranges"
    )

    st.caption(
        "These ranges control the dashboard's direct environmental "
        "warnings. They do not retrain or recalibrate the sensor model."
    )

    with st.form(
        "environment_settings_form"
    ):
        e1, e2 = st.columns(2)

        with e1:
            new_ph_min = st.number_input(
                "pH minimum",
                value=float(SETTINGS["ph_min"]),
                step=0.1,
            )
            new_tds_min = st.number_input(
                "TDS minimum (ppm)",
                value=float(SETTINGS["tds_min"]),
                step=10.0,
            )
            new_air_min = st.number_input(
                "Air temperature minimum (°C)",
                value=float(SETTINGS["air_temp_min"]),
                step=0.5,
            )
            new_humidity_min = st.number_input(
                "Humidity minimum (%)",
                value=float(SETTINGS["humidity_min"]),
                step=1.0,
            )
            new_water_temp_min = st.number_input(
                "Water temperature minimum (°C)",
                value=float(SETTINGS["water_temp_min"]),
                step=0.5,
            )

        with e2:
            new_ph_max = st.number_input(
                "pH maximum",
                value=float(SETTINGS["ph_max"]),
                step=0.1,
            )
            new_tds_max = st.number_input(
                "TDS maximum (ppm)",
                value=float(SETTINGS["tds_max"]),
                step=10.0,
            )
            new_air_max = st.number_input(
                "Air temperature maximum (°C)",
                value=float(SETTINGS["air_temp_max"]),
                step=0.5,
            )
            new_humidity_max = st.number_input(
                "Humidity maximum (%)",
                value=float(SETTINGS["humidity_max"]),
                step=1.0,
            )
            new_water_temp_max = st.number_input(
                "Water temperature maximum (°C)",
                value=float(SETTINGS["water_temp_max"]),
                step=0.5,
            )

        save_environment = st.form_submit_button(
            "💾 Save Environmental Ranges",
            use_container_width=True,
        )

    if save_environment:
        ranges_valid = all(
            [
                new_ph_min < new_ph_max,
                new_tds_min < new_tds_max,
                new_air_min < new_air_max,
                new_humidity_min < new_humidity_max,
                new_water_temp_min < new_water_temp_max,
            ]
        )

        if not ranges_valid:
            st.error(
                "Each minimum value must be lower than its maximum."
            )

        else:
            save_system_settings(
                {
                    "ph_min": new_ph_min,
                    "ph_max": new_ph_max,
                    "tds_min": new_tds_min,
                    "tds_max": new_tds_max,
                    "air_temp_min": new_air_min,
                    "air_temp_max": new_air_max,
                    "humidity_min": new_humidity_min,
                    "humidity_max": new_humidity_max,
                    "water_temp_min": new_water_temp_min,
                    "water_temp_max": new_water_temp_max,
                }
            )

            st.success(
                "Environmental reference ranges updated."
            )

            st.rerun()

    st.divider()

    st.markdown(
        "### Current Shared Settings"
    )

    current_settings_df = pd.DataFrame(
        [
            {
                "Setting": key,
                "Value": value,
            }
            for key, value in SETTINGS.items()
        ]
    )

    st.dataframe(
        current_settings_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE: HISTORY
# ============================================================

elif page == "History":

    st.markdown(
        "## Historical Monitoring"
    )

    st.caption(
        "Temporal sensor and camera-light measurements "
        "collected by the edge system."
    )

    history = get_sensor_history()

    if history.empty:
        st.info(
            "No historical data available."
        )

    else:
        range_option = st.selectbox(
            "Display period",
            [
                "Last 24 Hours",
                "Last 7 Days",
                "All Data",
            ],
        )

        filtered = history.copy()

        now = pd.Timestamp.now()

        if range_option == "Last 24 Hours":
            filtered = filtered[
                filtered["timestamp"]
                >= now
                - pd.Timedelta(
                    hours=24
                )
            ]

        elif range_option == "Last 7 Days":
            filtered = filtered[
                filtered["timestamp"]
                >= now
                - pd.Timedelta(
                    days=7
                )
            ]

        if filtered.empty:
            st.info(
                "No data for the selected period."
            )

        else:
            filtered = filtered.set_index(
                "timestamp"
            )

            st.markdown(
                "### pH"
            )
            st.line_chart(
                filtered[["ph"]]
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    "### EC"
                )
                st.line_chart(
                    filtered[["ec_ms_cm"]]
                )

            with c2:
                st.markdown(
                    "### TDS"
                )
                st.line_chart(
                    filtered[["tds_ppm"]]
                )

            st.markdown(
                "### Temperature"
            )
            st.line_chart(
                filtered[
                    [
                        "water_temp_c",
                        "air_temp_c",
                    ]
                ]
            )

            c3, c4 = st.columns(2)

            with c3:
                st.markdown(
                    "### Humidity"
                )
                st.line_chart(
                    filtered[
                        ["humidity_pct"]
                    ]
                )

            with c4:
                st.markdown(
                    "### Camera Light"
                )
                st.line_chart(
                    filtered[
                        ["camera_lux"]
                    ]
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Hydroponic Lettuce Edge-AI Monitoring System
        <br>
        Raspberry Pi • ESP32 • SQLite • MobileNetV2 •
        Random Forest • Complementary Decision Fusion •
        Grad-CAM • SHAP
    </div>
    """,
    unsafe_allow_html=True,
)
