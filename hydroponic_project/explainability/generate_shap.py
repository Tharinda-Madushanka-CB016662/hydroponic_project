#!/usr/bin/env python3
"""Final sensor SHAP explainability for Hydroponic Edge-AI.

One SHAP explanation is generated per capture because the same environmental
sensor snapshot is shared by all plant positions in that capture.

The final Random Forest uses exactly five features:
    pH, TDS, DHT_temp, DHT_humidity, water_temp

Water level is intentionally excluded from the model and remains a direct
operational alert.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB_FILE = Path('/hydroponic_project/database/hydroponic.db')

SENSOR_MODEL_CANDIDATES = [
    Path('/hydroponic_project/models/sensor/production/sensor_model_package_expert_5features.joblib'),
    Path('/hydroponic_project/models/sensor/candidates/sensor_model_package_expert_5features.joblib'),
]

SHAP_ROOT = Path('/hydroponic_project/explainability/shap')
DEFAULT_SENSOR_STRESS_THRESHOLD = 0.48

EXPECTED_SENSOR_FEATURES = [
    'pH',
    'TDS',
    'DHT_temp',
    'DHT_humidity',
    'water_temp',
]

FEATURE_DISPLAY_NAMES = {
    'pH': 'pH',
    'TDS': 'TDS',
    'DHT_temp': 'Air Temperature',
    'DHT_humidity': 'Humidity',
    'water_temp': 'Water Temperature',
}


def resolve_model_file():
    for path in SENSOR_MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        'Final expert 5-feature sensor model not found. Checked:\n'
        + '\n'.join(f'  - {path}' for path in SENSOR_MODEL_CANDIDATES)
    )


def check_database():
    if not DB_FILE.exists():
        raise FileNotFoundError(f'Database not found: {DB_FILE}')

    conn = sqlite3.connect(str(DB_FILE))
    try:
        columns = {
            row[1]
            for row in conn.execute('PRAGMA table_info(capture_events)').fetchall()
        }
        if 'shap_path' not in columns:
            raise RuntimeError(
                'capture_events does not contain shap_path column.'
            )
    finally:
        conn.close()


def get_sensor_threshold():
    conn = sqlite3.connect(str(DB_FILE))
    try:
        row = conn.execute(
            """
            SELECT setting_value
            FROM system_settings
            WHERE setting_key = 'sensor_stress_threshold'
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        conn.close()

    if row is None:
        return DEFAULT_SENSOR_STRESS_THRESHOLD
    return float(row[0])


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
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f'Capture not found: {capture_id}'
                if capture_id else 'No capture event found.'
            )
        return dict(row)
    finally:
        conn.close()


def load_sensor_model():
    model_file = resolve_model_file()
    start = time.perf_counter()
    package = joblib.load(model_file)
    model = package['model']
    feature_names = list(package.get('features', []))

    if feature_names != EXPECTED_SENSOR_FEATURES:
        raise RuntimeError(
            'Final sensor package feature mismatch.\n'
            f'Expected: {EXPECTED_SENSOR_FEATURES}\n'
            f'Loaded  : {feature_names}'
        )

    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise RuntimeError(f'Expected classes [0, 1], found {classes}')

    elapsed = (time.perf_counter() - start) * 1000

    print(f'Model file : {model_file}')
    print(f'Model      : {package.get("best_model_name")}')
    print(f'Features   : {feature_names}')
    print(f'Classes    : {classes}')
    print(f'Load time  : {elapsed:.2f} ms')

    return model, package


def build_sensor_dataframe(capture, feature_names):
    values = {
        'pH': float(capture['ph']),
        'TDS': float(capture['tds_ppm']),
        'DHT_temp': float(capture['air_temp_c']),
        'DHT_humidity': float(capture['humidity_pct']),
        'water_temp': float(capture['water_temp_c']),
    }

    missing = [name for name in feature_names if name not in values]
    if missing:
        raise RuntimeError(
            'Unable to map model feature(s): ' + ', '.join(missing)
        )

    dataframe = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names,
    )

    return dataframe, values


def extract_class_shap_values(
    raw_values,
    class_position,
    number_of_classes,
    number_of_features,
):
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[class_position])
        if values.ndim == 2:
            values = values[0]
        return values.astype(float)

    values = np.asarray(raw_values)

    if values.ndim == 3:
        if (
            values.shape[0] == 1
            and values.shape[1] == number_of_features
            and values.shape[2] == number_of_classes
        ):
            return values[0, :, class_position].astype(float)

        if (
            values.shape[0] == number_of_classes
            and values.shape[1] == 1
            and values.shape[2] == number_of_features
        ):
            return values[class_position, 0, :].astype(float)

    if values.ndim == 2 and values.shape == (1, number_of_features):
        result = values[0].astype(float)
        if number_of_classes == 2 and class_position == 0:
            result = -result
        return result

    if values.ndim == 1 and values.shape[0] == number_of_features:
        result = values.astype(float)
        if number_of_classes == 2 and class_position == 0:
            result = -result
        return result

    raise RuntimeError(f'Unsupported SHAP output shape: {values.shape}')


def calculate_shap(model, dataframe, explained_class):
    classes = list(model.classes_)
    class_position = classes.index(explained_class)
    explainer = shap.TreeExplainer(model)

    try:
        explanation = explainer(dataframe)
        raw_values = explanation.values
    except Exception:
        raw_values = explainer.shap_values(dataframe)

    values = extract_class_shap_values(
        raw_values,
        class_position,
        len(classes),
        dataframe.shape[1],
    )

    if len(values) != dataframe.shape[1]:
        raise RuntimeError(
            'SHAP feature-count mismatch. '
            f'Expected {dataframe.shape[1]}, received {len(values)}.'
        )

    return values


def format_feature_label(feature_name, value):
    display_name = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
    numeric_value = float(value)

    if feature_name == 'TDS':
        return f'{display_name} = {numeric_value:.1f} ppm'
    if feature_name in {'DHT_temp', 'water_temp'}:
        return f'{display_name} = {numeric_value:.2f} °C'
    if feature_name == 'DHT_humidity':
        return f'{display_name} = {numeric_value:.1f}%'
    return f'{display_name} = {numeric_value:.2f}'


def create_shap_chart(
    feature_names,
    feature_values,
    shap_values,
    decision_label,
    stress_probability,
    threshold,
    capture_id,
    output_path,
):
    values = np.asarray(shap_values, dtype=float)
    order = np.argsort(np.abs(values))
    ordered_values = values[order]
    ordered_feature_names = [feature_names[index] for index in order]

    labels = [
        format_feature_label(name, feature_values[name])
        for name in ordered_feature_names
    ]

    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    positions = np.arange(len(labels))
    axis.barh(positions, ordered_values, height=0.72)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=11)
    axis.tick_params(axis='y', pad=9)
    axis.axvline(0, linewidth=1.1)

    explained_direction = (
        'environmental stress class'
        if decision_label == 'Environmental Stress Warning'
        else 'healthy-environment class'
    )

    axis.set_xlabel(
        f"SHAP contribution toward the {explained_direction}",
        fontsize=11,
    )
    axis.set_title(
        'Random Forest Environmental Sensor Explanation\n'
        f'{decision_label} | Stress risk {stress_probability * 100:.2f}% '
        f'| Threshold {threshold:.2f}',
        fontsize=14,
        pad=12,
    )

    minimum_value = float(min(ordered_values.min(), 0.0))
    maximum_value = float(max(ordered_values.max(), 0.0))
    span = maximum_value - minimum_value
    if span <= 0:
        span = 1.0

    x_padding = span * 0.12
    axis.set_xlim(minimum_value - x_padding, maximum_value + x_padding)
    label_padding = span * 0.015

    for index, value in enumerate(ordered_values):
        axis.text(
            value + label_padding,
            index,
            f'{value:+.4f}',
            va='center',
            ha='left',
            fontsize=10,
            clip_on=True,
            bbox={
                'facecolor': 'white',
                'alpha': 0.72,
                'edgecolor': 'none',
                'pad': 1.5,
            },
        )

    axis.grid(axis='x', alpha=0.20)
    figure.text(
        0.5,
        0.025,
        f'Capture: {capture_id}',
        ha='center',
        fontsize=9,
    )
    figure.subplots_adjust(left=0.33, right=0.96, top=0.84, bottom=0.16)
    figure.savefig(output_path, dpi=170, bbox_inches='tight', pad_inches=0.25)
    plt.close(figure)


def update_database(capture_id, shap_path):
    conn = sqlite3.connect(str(DB_FILE), timeout=30)
    try:
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE capture_events SET shap_path = ? WHERE capture_id = ?',
            (str(shap_path), capture_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f'Unexpected capture update count: {cursor.rowcount}'
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--capture-id',
        default=None,
        help='Capture ID to explain. Default: latest capture.',
    )
    args = parser.parse_args()

    print('\n========================================')
    print(' FINAL SENSOR SHAP EXPLAINABILITY')
    print('========================================')

    check_database()
    threshold = get_sensor_threshold()
    capture = get_capture(args.capture_id)
    capture_id = capture['capture_id']

    print(f'Capture ID : {capture_id}')
    print(f'Timestamp  : {capture["timestamp"]}')
    print(f'Threshold  : {threshold:.2f}')

    model, package = load_sensor_model()
    feature_names = list(package['features'])
    dataframe, feature_value_map = build_sensor_dataframe(
        capture,
        feature_names,
    )

    print('\nSensor input:')
    for feature in feature_names:
        print(f'  {feature:<15}: {feature_value_map[feature]}')
    print('  water_level    : EXCLUDED FROM RANDOM FOREST')

    probabilities = model.predict_proba(dataframe)[0]
    classes = list(model.classes_)
    stress_probability = float(probabilities[classes.index(0)])
    healthy_probability = float(probabilities[classes.index(1)])

    if stress_probability >= threshold:
        decision_label = 'Environmental Stress Warning'
        explained_class = 0
    else:
        decision_label = 'Healthy Environment'
        explained_class = 1

    print('\nPrediction:')
    print(f'  Decision     : {decision_label}')
    print(f'  Stress risk  : {stress_probability * 100:.2f}%')
    print(f'  Healthy prob.: {healthy_probability * 100:.2f}%')

    start = time.perf_counter()
    shap_values = calculate_shap(
        model,
        dataframe,
        explained_class,
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f'  SHAP time    : {elapsed:.2f} ms')

    output_dir = SHAP_ROOT / capture_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sensor_shap.png'

    create_shap_chart(
        feature_names,
        feature_value_map,
        shap_values,
        decision_label,
        stress_probability,
        threshold,
        capture_id,
        output_path,
    )

    update_database(capture_id, output_path)

    print('\nSHAP feature contributions:')
    rows = sorted(
        zip(feature_names, shap_values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    for feature, value in rows:
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        print(f'  {display_name:<20}{float(value):+.6f}')

    print('\n========================================')
    print(' SHAP COMPLETED')
    print('========================================')
    print(f'Decision   : {decision_label}')
    print(f'Stress risk: {stress_probability * 100:.2f}%')
    print(f'SHAP chart : {output_path}')
    print(f'Database   : {DB_FILE}')
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('\n========================================')
        print(' SHAP FAILED')
        print('========================================')
        print(f'{type(error).__name__}: {error}')
        sys.exit(1)
