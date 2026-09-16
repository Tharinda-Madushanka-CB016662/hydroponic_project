#!/usr/bin/env python3
"""Final Hydroponic Edge-AI inference pipeline.

Final deployment design
-----------------------
1. New MobileNetV2 -> current visible plant condition.
2. New expert-trained 5-feature Random Forest -> current environmental stress risk.
3. Complementary decision-level fusion -> preserves both meanings instead of
   averaging probabilities into one binary health label.
4. Water level remains an independent operational alert and is NOT a Random
   Forest input feature.

Historical classical-fusion database columns are preserved for research audit.
This script deliberately DOES NOT overwrite fusion_prediction or
fusion_confidence. New captures leave those legacy fields NULL, while old
captures retain their historical classical-fusion results.
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path('/hydroponic_project/database/hydroponic.db')

IMAGE_MODEL_CANDIDATES = [
    Path('/hydroponic_project/models/image/production/mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth'),
]

SENSOR_MODEL_CANDIDATES = [
    Path('/hydroponic_project/models/sensor/production/sensor_model_package_expert_5features.joblib'),
]


# ============================================================
# FINAL DEFAULT SETTINGS
# ============================================================

DEFAULT_SETTINGS = {
    'min_image_lux': 10.0,
    'image_start_hour': 6,
    'image_end_hour': 17,
    'image_unhealthy_threshold': 0.42,
    'sensor_stress_threshold': 0.48,
}

EXPECTED_SENSOR_FEATURES = [
    'pH',
    'TDS',
    'DHT_temp',
    'DHT_humidity',
    'water_temp',
]

DEVICE = torch.device('cpu')
torch.set_num_threads(4)


# ============================================================
# MODEL WRAPPER / PREPROCESSING
# ============================================================

class Model2Class(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# HELPERS
# ============================================================

def resolve_required_file(candidates, description):
    for candidate in candidates:
        if candidate.exists():
            return candidate

    paths = '\n'.join(f'  - {path}' for path in candidates)
    raise FileNotFoundError(
        f'{description} not found. Checked:\n{paths}'
    )


def table_columns(conn, table_name):
    return {
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info({table_name})'
        ).fetchall()
    }


def ensure_final_schema():
    """Add final-deployment columns/settings without deleting legacy data."""

    if not DB_FILE.exists():
        raise FileNotFoundError(f'Database not found: {DB_FILE}')

    conn = sqlite3.connect(str(DB_FILE), timeout=30)

    try:
        required_tables = {'capture_events', 'plant_samples'}
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        missing_tables = required_tables - tables
        if missing_tables:
            raise RuntimeError(
                'Missing table(s): ' + ', '.join(sorted(missing_tables))
            )

        plant_columns = table_columns(conn, 'plant_samples')
        for name, sql_type in {
            'image_unhealthy_probability': 'REAL',
            'sensor_stress_probability': 'REAL',
            'decision_state': 'TEXT',
        }.items():
            if name not in plant_columns:
                conn.execute(
                    f'ALTER TABLE plant_samples ADD COLUMN {name} {sql_type}'
                )

        capture_columns = table_columns(conn, 'capture_events')
        if 'environmental_warning' not in capture_columns:
            conn.execute(
                'ALTER TABLE capture_events ADD COLUMN environmental_warning INTEGER'
            )

        conn.execute(
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
            'min_image_lux': 'Minimum camera Lux required for image AI',
            'image_start_hour': 'Image inference start hour',
            'image_end_hour': 'Image inference end hour, exclusive',
            'image_unhealthy_threshold': 'Final MobileNetV2 unhealthy probability threshold',
            'sensor_stress_threshold': 'Final 5-feature RF environmental-stress probability threshold',
        }

        for key, value in DEFAULT_SETTINGS.items():
            conn.execute(
                """
                INSERT OR IGNORE INTO system_settings (
                    setting_key,
                    setting_value,
                    description
                ) VALUES (?, ?, ?)
                """,
                (key, str(value), descriptions[key]),
            )

        conn.commit()

    finally:
        conn.close()


def load_system_settings():
    conn = sqlite3.connect(str(DB_FILE))

    try:
        rows = conn.execute(
            """
            SELECT setting_key, setting_value
            FROM system_settings
            WHERE setting_key IN (
                'min_image_lux',
                'image_start_hour',
                'image_end_hour',
                'image_unhealthy_threshold',
                'sensor_stress_threshold'
            )
            """
        ).fetchall()

    finally:
        conn.close()

    raw = dict(rows)

    return {
        'min_image_lux': float(raw.get('min_image_lux', DEFAULT_SETTINGS['min_image_lux'])),
        'image_start_hour': int(float(raw.get('image_start_hour', DEFAULT_SETTINGS['image_start_hour']))),
        'image_end_hour': int(float(raw.get('image_end_hour', DEFAULT_SETTINGS['image_end_hour']))),
        'image_unhealthy_threshold': float(raw.get('image_unhealthy_threshold', DEFAULT_SETTINGS['image_unhealthy_threshold'])),
        'sensor_stress_threshold': float(raw.get('sensor_stress_threshold', DEFAULT_SETTINGS['sensor_stress_threshold'])),
    }


def check_image_suitability(timestamp, camera_lux, settings):
    capture_time = datetime.fromisoformat(timestamp)
    hour = capture_time.hour

    inside_time_window = (
        settings['image_start_hour']
        <= hour
        < settings['image_end_hour']
    )

    if not inside_time_window:
        return {
            'allowed': False,
            'reason': 'OUTSIDE TIME WINDOW',
            'capture_time': capture_time,
            'lux': camera_lux,
        }

    if camera_lux is None:
        return {
            'allowed': False,
            'reason': 'CAMERA LUX UNAVAILABLE',
            'capture_time': capture_time,
            'lux': None,
        }

    try:
        lux = float(camera_lux)
    except (TypeError, ValueError):
        return {
            'allowed': False,
            'reason': 'INVALID CAMERA LUX',
            'capture_time': capture_time,
            'lux': camera_lux,
        }

    if lux < settings['min_image_lux']:
        return {
            'allowed': False,
            'reason': 'LOW LIGHT',
            'capture_time': capture_time,
            'lux': lux,
        }

    return {
        'allowed': True,
        'reason': 'IMAGE SUITABLE',
        'capture_time': capture_time,
        'lux': lux,
    }


def water_level_is_low(value):
    text = str(value or '').strip().upper()
    return text in {
        'LOW',
        'LOW ALERT',
        'ALERT',
        'EMPTY',
        '0',
        'FALSE',
    }


# ============================================================
# MODEL LOADERS
# ============================================================

def load_image_model():
    model_file = resolve_required_file(
        IMAGE_MODEL_CANDIDATES,
        'Final MobileNetV2 checkpoint',
    )

    print('\n========================================')
    print('Loading final MobileNetV2...')
    print('========================================')

    start = time.perf_counter()

    checkpoint = torch.load(
        model_file,
        map_location=DEVICE,
        weights_only=False,
    )

    model = Model2Class(
        num_classes=checkpoint.get('num_classes', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    class_names = list(checkpoint['class_names'])
    class_to_idx = dict(checkpoint['class_to_idx'])

    if class_to_idx != {'healthy': 0, 'unhealthy': 1}:
        raise RuntimeError(
            f'Unexpected image class mapping: {class_to_idx}'
        )

    elapsed = (time.perf_counter() - start) * 1000

    print(f'Checkpoint   : {model_file}')
    print(f'Model        : {checkpoint.get("model_name")}')
    print(f'Classes      : {class_names}')
    print(f'Class map    : {class_to_idx}')
    print(f'Best epoch   : {checkpoint.get("best_epoch")}')
    print(f'Best val F1  : {checkpoint.get("best_val_macro_f1")}')
    print(f'Load time    : {elapsed:.2f} ms')

    return model, class_names, class_to_idx


def load_sensor_model():
    model_file = resolve_required_file(
        SENSOR_MODEL_CANDIDATES,
        'Final expert 5-feature Random Forest package',
    )

    print('\n========================================')
    print('Loading final 5-feature sensor RF...')
    print('========================================')

    start = time.perf_counter()
    package = joblib.load(model_file)
    model = package['model']
    features = list(package.get('features', []))

    if features != EXPECTED_SENSOR_FEATURES:
        raise RuntimeError(
            'Final sensor model must use exactly these 5 features:\n'
            f'{EXPECTED_SENSOR_FEATURES}\n'
            f'Loaded package uses:\n{features}'
        )

    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise RuntimeError(
            f'Expected sensor classes [0, 1], found {classes}'
        )

    elapsed = (time.perf_counter() - start) * 1000

    print(f'Package      : {model_file}')
    print(f'Model        : {package.get("best_model_name")}')
    print(f'Features     : {features}')
    print(f'Classes      : {classes}')
    print(f'Class meaning: 0=Environmental Stress, 1=Healthy Environment')
    print(f'Load time    : {elapsed:.2f} ms')

    return model, package


# ============================================================
# DATABASE CAPTURE LOAD
# ============================================================

def get_capture(capture_id=None):
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        if capture_id:
            cursor.execute(
                'SELECT * FROM capture_events WHERE capture_id = ?',
                (capture_id,),
            )
        else:
            cursor.execute(
                'SELECT * FROM capture_events ORDER BY id DESC LIMIT 1'
            )

        capture = cursor.fetchone()

        if capture is None:
            raise RuntimeError(
                f'Capture not found: {capture_id}'
                if capture_id
                else 'No capture event found.'
            )

        actual_capture_id = capture['capture_id']

        cursor.execute(
            """
            SELECT
                id,
                capture_id,
                plant_id,
                crop_path,
                model_image_path,
                plant_active
            FROM plant_samples
            WHERE capture_id = ?
            ORDER BY plant_id
            """,
            (actual_capture_id,),
        )

        plants = [dict(row) for row in cursor.fetchall()]

        if not plants:
            raise RuntimeError(
                f'No plant samples found for {actual_capture_id}'
            )

        return dict(capture), plants

    finally:
        conn.close()


# ============================================================
# SENSOR INFERENCE
# ============================================================

def predict_sensor(sensor_model, sensor_package, capture, threshold):
    """Predict environmental suitability using exactly five sensor features."""

    feature_map = {
        'pH': float(capture['ph']),
        'TDS': float(capture['tds_ppm']),
        'DHT_temp': float(capture['air_temp_c']),
        'DHT_humidity': float(capture['humidity_pct']),
        'water_temp': float(capture['water_temp_c']),
    }

    feature_names = list(sensor_package['features'])

    # water_level is intentionally NOT present here.
    dataframe = pd.DataFrame(
        [[feature_map[name] for name in feature_names]],
        columns=feature_names,
    )

    start = time.perf_counter()
    probabilities = sensor_model.predict_proba(dataframe)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    classes = list(sensor_model.classes_)
    stress_index = classes.index(0)
    healthy_index = classes.index(1)

    stress_probability = float(probabilities[stress_index])
    healthy_probability = float(probabilities[healthy_index])

    # Deployment decision uses the validated sensor threshold (0.48),
    # not simple argmax/0.50.
    if stress_probability >= threshold:
        prediction = 'Environmental Stress Warning'
        confidence = stress_probability
        prediction_code = 'stress'
    else:
        prediction = 'Healthy Environment'
        confidence = healthy_probability
        prediction_code = 'healthy'

    return {
        'prediction': prediction,
        'prediction_code': prediction_code,
        'confidence': float(confidence),
        'stress_probability': stress_probability,
        'healthy_probability': healthy_probability,
        'inference_ms': inference_ms,
        'features': feature_map,
    }


# ============================================================
# IMAGE INFERENCE
# ============================================================

def predict_image(
    model,
    image_path,
    class_names,
    class_to_idx,
    threshold,
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f'Plant image missing: {image_path}')

    image = Image.open(image_path).convert('RGB')
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    inference_ms = (time.perf_counter() - start) * 1000

    unhealthy_index = class_to_idx['unhealthy']
    healthy_index = class_to_idx['healthy']

    unhealthy_probability = float(probabilities[unhealthy_index])
    healthy_probability = float(probabilities[healthy_index])

    # Deployment decision uses validated live threshold 0.42.
    if unhealthy_probability >= threshold:
        prediction = 'unhealthy'
        confidence = unhealthy_probability
    else:
        prediction = 'healthy'
        confidence = healthy_probability

    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'unhealthy_probability': unhealthy_probability,
        'healthy_probability': healthy_probability,
        'inference_ms': inference_ms,
    }


# ============================================================
# COMPLEMENTARY DECISION-LEVEL FUSION
# ============================================================

def complementary_decision(image_prediction, sensor_prediction_code):
    """Preserve visible-health and environmental-risk meanings."""

    image_unhealthy = str(image_prediction).strip().lower() == 'unhealthy'
    sensor_stress = sensor_prediction_code == 'stress'

    if not image_unhealthy and not sensor_stress:
        return 'HEALTHY / STABLE'

    if not image_unhealthy and sensor_stress:
        return 'EARLY ENVIRONMENTAL WARNING'

    if image_unhealthy and not sensor_stress:
        return 'VISIBLE PLANT STRESS - ENVIRONMENT CURRENTLY NORMAL'

    return 'HIGH RISK - VISIBLE + ENVIRONMENTAL STRESS'


# ============================================================
# RESULT BUILDERS
# ============================================================

def create_full_result(plant, image_result, sensor_result):
    decision_state = complementary_decision(
        image_result['prediction'],
        sensor_result['prediction_code'],
    )

    return {
        'plant_db_id': plant['id'],
        'image_prediction': image_result['prediction'],
        'image_confidence': image_result['confidence'],
        'image_unhealthy_probability': image_result['unhealthy_probability'],
        'sensor_prediction': sensor_result['prediction'],
        'sensor_confidence': sensor_result['confidence'],
        'sensor_stress_probability': sensor_result['stress_probability'],
        'decision_state': decision_state,
        'final_status': decision_state,
        # Classical probability fusion is intentionally retired for new data.
        'fusion_prediction': None,
        'fusion_confidence': None,
    }


def create_sensor_only_result(plant, sensor_result, skip_reason):
    if sensor_result['prediction_code'] == 'stress':
        final_status = 'ENVIRONMENTAL STRESS WARNING - IMAGE NOT EVALUATED'
    else:
        final_status = 'ENVIRONMENT NORMAL - IMAGE NOT EVALUATED'

    return {
        'plant_db_id': plant['id'],
        'image_prediction': f'SKIPPED - {skip_reason}',
        'image_confidence': None,
        'image_unhealthy_probability': None,
        'sensor_prediction': sensor_result['prediction'],
        'sensor_confidence': sensor_result['confidence'],
        'sensor_stress_probability': sensor_result['stress_probability'],
        'decision_state': None,
        'final_status': final_status,
        'fusion_prediction': None,
        'fusion_confidence': None,
    }


def create_no_plant_result(plant):
    return {
        'plant_db_id': plant['id'],
        'image_prediction': 'NO PLANT',
        'image_confidence': None,
        'image_unhealthy_probability': None,
        'sensor_prediction': 'NOT APPLICABLE',
        'sensor_confidence': None,
        'sensor_stress_probability': None,
        'decision_state': None,
        'final_status': 'NO PLANT',
        'fusion_prediction': None,
        'fusion_confidence': None,
    }


def create_unknown_result(plant):
    return {
        'plant_db_id': plant['id'],
        'image_prediction': 'OCCUPANCY UNKNOWN',
        'image_confidence': None,
        'image_unhealthy_probability': None,
        'sensor_prediction': 'NOT APPLICABLE',
        'sensor_confidence': None,
        'sensor_stress_probability': None,
        'decision_state': None,
        'final_status': 'OCCUPANCY UNKNOWN',
        'fusion_prediction': None,
        'fusion_confidence': None,
    }


# ============================================================
# SQLITE UPDATE
# ============================================================

def update_database(capture_id, results, sensor_result):
    conn = sqlite3.connect(str(DB_FILE), timeout=30)

    try:
        cursor = conn.cursor()

        for result in results:
            cursor.execute(
                """
                UPDATE plant_samples
                SET
                    image_prediction = ?,
                    image_confidence = ?,
                    image_unhealthy_probability = ?,
                    sensor_prediction = ?,
                    sensor_confidence = ?,
                    sensor_stress_probability = ?,
                    decision_state = ?,
                    final_status = ?
                WHERE id = ?
                """,
                (
                    result['image_prediction'],
                    result['image_confidence'],
                    result['image_unhealthy_probability'],
                    result['sensor_prediction'],
                    result['sensor_confidence'],
                    result['sensor_stress_probability'],
                    result['decision_state'],
                    result['final_status'],
                    result['plant_db_id'],
                ),
            )

        cursor.execute(
            """
            UPDATE capture_events
            SET environmental_warning = ?
            WHERE capture_id = ?
            """,
            (
                1 if sensor_result['prediction_code'] == 'stress' else 0,
                capture_id,
            ),
        )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--capture-id',
        default=None,
        help='Capture ID to process. Default: latest capture.',
    )

    # Controlled experiment overrides only. SQLite settings are preferred.
    parser.add_argument('--image-threshold', type=float, default=None)
    parser.add_argument('--sensor-threshold', type=float, default=None)
    parser.add_argument('--min-lux', type=float, default=None)

    args = parser.parse_args()

    ensure_final_schema()
    settings = load_system_settings()

    if args.image_threshold is not None:
        settings['image_unhealthy_threshold'] = args.image_threshold

    if args.sensor_threshold is not None:
        settings['sensor_stress_threshold'] = args.sensor_threshold

    if args.min_lux is not None:
        settings['min_image_lux'] = args.min_lux

    for key in ['image_unhealthy_threshold', 'sensor_stress_threshold']:
        if not 0.0 <= settings[key] <= 1.0:
            raise ValueError(f'{key} must be between 0 and 1')

    if settings['image_end_hour'] <= settings['image_start_hour']:
        raise ValueError('image_end_hour must be greater than image_start_hour')

    total_start = time.perf_counter()

    print('\n========================================')
    print(' HYDROPONIC EDGE-AI FINAL INFERENCE')
    print('========================================')
    print(f'Device                  : {DEVICE}')
    print(f'Image unhealthy threshold: {settings["image_unhealthy_threshold"]:.2f}')
    print(f'Sensor stress threshold  : {settings["sensor_stress_threshold"]:.2f}')
    print('Decision strategy         : COMPLEMENTARY FOUR-STATE')
    print('Classical 60/40 fusion    : RETIRED FROM PRODUCTION')
    print(
        f'Image time window         : '
        f'{settings["image_start_hour"]:02d}:00 - '
        f'{settings["image_end_hour"] - 1:02d}:59'
    )
    print(f'Minimum Lux               : {settings["min_image_lux"]:.2f}')

    capture, plants = get_capture(args.capture_id)

    print('\n========================================')
    print('CAPTURE')
    print('========================================')
    print(f'Capture ID : {capture["capture_id"]}')
    print(f'Timestamp  : {capture["timestamp"]}')
    print(f'Camera Lux : {capture.get("camera_lux")}')
    print(f'Water level: {capture.get("water_level")}')

    active_count = sum(1 for plant in plants if plant['plant_active'] == 1)
    inactive_count = sum(1 for plant in plants if plant['plant_active'] == 0)
    unknown_count = sum(1 for plant in plants if plant['plant_active'] is None)

    print(f'Positions  : {len(plants)}')
    print(f'Active     : {active_count}')
    print(f'No plant   : {inactive_count}')
    print(f'Unknown    : {unknown_count}')

    if water_level_is_low(capture.get('water_level')):
        print('Operational alert: LOW WATER')
    else:
        print('Operational alert: Water level OK')

    # Sensor branch always runs.
    sensor_model, sensor_package = load_sensor_model()
    sensor_result = predict_sensor(
        sensor_model,
        sensor_package,
        capture,
        settings['sensor_stress_threshold'],
    )

    print('\n========================================')
    print('SENSOR ENVIRONMENTAL AI')
    print('========================================')
    print(f'Prediction       : {sensor_result["prediction"]}')
    print(f'Stress risk      : {sensor_result["stress_probability"] * 100:.2f}%')
    print(f'Healthy prob.    : {sensor_result["healthy_probability"] * 100:.2f}%')
    print(f'Decision threshold: {settings["sensor_stress_threshold"]:.2f}')
    print(f'Latency          : {sensor_result["inference_ms"]:.3f} ms')
    print('Features         : pH, TDS, air temp, humidity, water temp')
    print('Water level      : excluded from RF; handled as direct alert')

    image_quality = check_image_suitability(
        capture['timestamp'],
        capture.get('camera_lux'),
        settings,
    )

    print('\n========================================')
    print('IMAGE QUALITY CHECK')
    print('========================================')
    print(f'Capture time : {image_quality["capture_time"].strftime("%H:%M:%S")}')
    print(f'Camera Lux   : {image_quality["lux"]}')
    print(f'Minimum Lux  : {settings["min_image_lux"]}')
    print(f'Image status : {image_quality["reason"]}')

    image_model = None
    image_class_names = None
    image_class_to_idx = None

    if image_quality['allowed'] and active_count > 0:
        (
            image_model,
            image_class_names,
            image_class_to_idx,
        ) = load_image_model()
    else:
        print('\nImage AI will not run for this capture.')
        print('Sensor environmental inference remains valid.')
        print('Four-state decision is unavailable without image evidence.')

    results = []
    image_times = []
    full_decision_count = 0
    sensor_only_count = 0
    no_plant_count = 0
    unknown_processed = 0

    print('\n========================================')
    print('PLANT INFERENCE')
    print('========================================')

    for plant in plants:
        plant_id = plant['plant_id']
        plant_active = plant['plant_active']

        if plant_active == 0:
            print(f'\n{plant_id}: NO PLANT - AI skipped')
            results.append(create_no_plant_result(plant))
            no_plant_count += 1
            continue

        if plant_active is None:
            print(f'\n{plant_id}: OCCUPANCY UNKNOWN - AI skipped')
            results.append(create_unknown_result(plant))
            unknown_processed += 1
            continue

        if not image_quality['allowed']:
            skip_reason = image_quality['reason']
            result = create_sensor_only_result(
                plant,
                sensor_result,
                skip_reason,
            )
            results.append(result)
            sensor_only_count += 1

            print(f'\n{plant_id}: ACTIVE')
            print(f'  Image AI     : SKIPPED - {skip_reason}')
            print(f'  Sensor AI    : {sensor_result["prediction"]}')
            print(f'  Sensor risk  : {sensor_result["stress_probability"] * 100:.2f}%')
            print(f'  Final        : {result["final_status"]}')
            continue

        image_result = predict_image(
            image_model,
            plant['model_image_path'],
            image_class_names,
            image_class_to_idx,
            settings['image_unhealthy_threshold'],
        )

        image_times.append(image_result['inference_ms'])
        result = create_full_result(
            plant,
            image_result,
            sensor_result,
        )
        results.append(result)
        full_decision_count += 1

        print(f'\n{plant_id}: ACTIVE')
        print(f'  Image AI     : {image_result["prediction"]}')
        print(f'  Image risk   : {image_result["unhealthy_probability"] * 100:.2f}%')
        print(f'  Sensor AI    : {sensor_result["prediction"]}')
        print(f'  Sensor risk  : {sensor_result["stress_probability"] * 100:.2f}%')
        print(f'  Decision     : {result["decision_state"]}')
        print(f'  Image time   : {image_result["inference_ms"]:.2f} ms')

    update_database(
        capture['capture_id'],
        results,
        sensor_result,
    )

    total_ms = (time.perf_counter() - total_start) * 1000

    print('\n========================================')
    print(' INFERENCE COMPLETED')
    print('========================================')
    print(f'Capture ID            : {capture["capture_id"]}')
    print(f'Positions checked     : {len(plants)}')
    print(f'Four-state decisions  : {full_decision_count}')
    print(f'Sensor-only results   : {sensor_only_count}')
    print(f'No plant              : {no_plant_count}')
    print(f'Unknown occupancy     : {unknown_processed}')
    print(f'Image quality         : {image_quality["reason"]}')
    print(f'Sensor latency        : {sensor_result["inference_ms"]:.3f} ms')

    if image_times:
        print(f'Average image time    : {np.mean(image_times):.2f} ms')
        print(f'Minimum image time    : {np.min(image_times):.2f} ms')
        print(f'Maximum image time    : {np.max(image_times):.2f} ms')
    else:
        print('Image inference       : Not performed')

    print(f'Total script time     : {total_ms:.2f} ms')
    print(f'Predictions saved to  : {DB_FILE}')
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('\n========================================')
        print(' INFERENCE FAILED')
        print('========================================')
        print(f'{type(error).__name__}: {error}')
        sys.exit(1)
