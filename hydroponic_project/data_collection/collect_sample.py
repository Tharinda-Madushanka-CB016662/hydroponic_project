import cv2
import csv
import json
import serial
import sqlite3
import sys

from datetime import datetime
from pathlib import Path
from time import sleep

from picamera2 import Picamera2


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/hydroponic_project/data_collection"
)

IMAGE_DIR = BASE_DIR / "images"

CROP_ROOT = BASE_DIR / "crops"

ROI_FILE = Path(
    "/hydroponic_project/image_data/config/plant_rois.json"
)

CSV_FILE = (
    BASE_DIR /
    "multimodal_dataset.csv"
)

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_WIDTH = 3280

CAMERA_HEIGHT = 2464

JPEG_QUALITY = 95


# ============================================================
# ESP32 SERIAL
# ============================================================

SERIAL_PORT = "/dev/ttyUSB0"

BAUD_RATE = 115200


# ============================================================
# PLANTS
# ============================================================

NUMBER_OF_PLANTS = 12


# ============================================================
# CREATE DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CROP_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK DATABASE
# ============================================================

def check_database():

    if not DB_FILE.exists():

        raise FileNotFoundError(
            f"SQLite database not found: "
            f"{DB_FILE}"
        )

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    try:

        cursor = conn.cursor()

        # ----------------------------------------------------
        # Check required tables
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """
        )

        tables = {
            row[0]
            for row in cursor.fetchall()
        }

        required_tables = {
            "capture_events",
            "plant_samples",
            "plant_positions"
        }

        missing_tables = (
            required_tables - tables
        )

        if missing_tables:

            raise RuntimeError(
                "Missing SQLite table(s): "
                + ", ".join(
                    sorted(
                        missing_tables
                    )
                )
            )

        # ----------------------------------------------------
        # Check plant_active column
        # ----------------------------------------------------

        cursor.execute(
            """
            PRAGMA table_info(
                plant_samples
            )
            """
        )

        columns = {
            row[1]
            for row in cursor.fetchall()
        }

        if "plant_active" not in columns:

            raise RuntimeError(
                "plant_samples table does not "
                "contain plant_active column."
            )

        # ----------------------------------------------------
        # Check plant positions
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                plant_id,
                active
            FROM plant_positions
            ORDER BY plant_id
            """
        )

        positions = cursor.fetchall()

        if len(positions) != NUMBER_OF_PLANTS:

            print()
            print(
                "WARNING:"
            )

            print(
                f"Expected {NUMBER_OF_PLANTS} "
                f"plant positions but found "
                f"{len(positions)}."
            )

    finally:

        conn.close()


# ============================================================
# GET CURRENT PLANT ACTIVITY
# ============================================================

def get_plant_activity():

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                plant_id,
                active
            FROM plant_positions
            ORDER BY plant_id
            """
        )

        rows = cursor.fetchall()

        activity = {

            plant_id:
                int(active)

            for plant_id, active
            in rows
        }

        return activity

    finally:

        conn.close()


# ============================================================
# DISPLAY CURRENT PLANT POSITIONS
# ============================================================

def display_plant_activity(
    plant_activity
):

    print()
    print("========================================")
    print(" CURRENT PLANT POSITIONS")
    print("========================================")

    active_count = 0

    for number in range(
        1,
        NUMBER_OF_PLANTS + 1
    ):

        plant_id = (
            f"P{number:02d}"
        )

        active = (
            plant_activity.get(
                plant_id,
                0
            )
        )

        if active == 1:

            status = "ACTIVE"

            active_count += 1

        else:

            status = "NO PLANT"

        print(
            f"{plant_id}: {status}"
        )

    print()

    print(
        f"Active plants: "
        f"{active_count}/"
        f"{NUMBER_OF_PLANTS}"
    )


# ============================================================
# READ ONE VALID SENSOR RECORD
# ============================================================

def read_sensor_record():

    print()
    print("========================================")
    print("Reading ESP32 sensors...")
    print("========================================")

    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=3
    )

    # ESP32 may reset when USB serial opens
    sleep(2)

    ser.reset_input_buffer()

    try:

        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:

                continue

            if not line.startswith("{"):

                continue

            if not line.endswith("}"):

                continue

            try:

                data = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            # Ignore incomplete / status records
            if "water_temp" not in data:

                continue

            print()
            print(
                "Sensor record received:"
            )

            print(
                json.dumps(
                    data,
                    indent=2
                )
            )

            return data

    finally:

        ser.close()


# ============================================================
# CAPTURE CAMERA IMAGE
# ============================================================

def capture_image(
    timestamp_file
):

    print()
    print("========================================")
    print("Capturing camera image...")
    print("========================================")

    camera = Picamera2()

    config = (
        camera.create_still_configuration(

            main={
                "size": (
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT
                )
            },

            queue=False
        )
    )

    camera.configure(
        config
    )

    camera.options[
        "quality"
    ] = JPEG_QUALITY

    camera.set_controls({

        "AeEnable":
            True,

        "AwbEnable":
            True,

        "Brightness":
            0.0,

        "Contrast":
            1.0,

        "Sharpness":
            1.0,

        "Saturation":
            1.0
    })

    camera.start()

    print(
        "Waiting for exposure and "
        "white balance..."
    )

    sleep(10)

    image_path = (

        IMAGE_DIR /

        f"lettuce_"
        f"{timestamp_file}.jpg"
    )

    request = None

    try:

        request = (
            camera.capture_request(
                flush=True
            )
        )

        request.save(
            "main",
            str(image_path)
        )

        camera_metadata = (
            request.get_metadata()
        )

    finally:

        if request is not None:

            request.release()

        camera.stop()

        camera.close()

    print()
    print(
        "Image saved:"
    )

    print(
        image_path
    )

    print()
    print(
        "Camera metadata:"
    )

    print(
        "ExposureTime:",
        camera_metadata.get(
            "ExposureTime"
        )
    )

    print(
        "AnalogueGain:",
        camera_metadata.get(
            "AnalogueGain"
        )
    )

    print(
        "DigitalGain:",
        camera_metadata.get(
            "DigitalGain"
        )
    )

    print(
        "Lux:",
        camera_metadata.get(
            "Lux"
        )
    )

    return (
        image_path,
        camera_metadata
    )


# ============================================================
# CROP ALL 12 PLANTS
# ============================================================

def crop_plants(
    image_path,
    timestamp_file
):

    print()
    print("========================================")
    print("Cropping P01-P12...")
    print("========================================")

    if not ROI_FILE.exists():

        raise FileNotFoundError(
            f"ROI file missing: "
            f"{ROI_FILE}"
        )

    with open(
        ROI_FILE,
        "r"
    ) as file:

        rois = json.load(
            file
        )

    if (
        len(rois)
        != NUMBER_OF_PLANTS
    ):

        print(
            f"WARNING: Expected "
            f"{NUMBER_OF_PLANTS} ROIs, "
            f"but found {len(rois)}."
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            "Unable to load "
            "captured image."
        )

    (
        image_height,
        image_width

    ) = image.shape[:2]

    output_dir = (

        CROP_ROOT /

        f"lettuce_"
        f"{timestamp_file}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    crop_records = []

    for roi in rois:

        plant_id = (
            roi[
                "plant_id"
            ]
        )

        x = int(
            roi["x"]
        )

        y = int(
            roi["y"]
        )

        w = int(
            roi["w"]
        )

        h = int(
            roi["h"]
        )

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            image_width,
            x + w
        )

        y2 = min(
            image_height,
            y + h
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            print(
                f"Skipping empty crop: "
                f"{plant_id}"
            )

            continue

        # ----------------------------------------------------
        # Original crop
        # ----------------------------------------------------

        crop_path = (

            output_dir /

            f"{plant_id}.jpg"
        )

        cv2.imwrite(

            str(
                crop_path
            ),

            crop,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        # ----------------------------------------------------
        # Resize to 224 x 224 for MobileNetV2
        # ----------------------------------------------------

        resized = cv2.resize(

            crop,

            (
                224,
                224
            ),

            interpolation=
                cv2.INTER_AREA
        )

        model_path = (

            output_dir /

            f"{plant_id}_224.jpg"
        )

        cv2.imwrite(

            str(
                model_path
            ),

            resized,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        crop_records.append({

            "plant_id":
                plant_id,

            "crop_path":
                str(
                    crop_path
                ),

            "model_image_path":
                str(
                    model_path
                )
        })

        print(
            f"{plant_id}: "
            f"{crop.shape[1]}x"
            f"{crop.shape[0]} "
            f"-> 224x224"
        )

    print()

    print(
        f"{len(crop_records)} "
        f"plant crops created."
    )

    return crop_records


# ============================================================
# SAVE TO SQLITE DATABASE
# ============================================================

def save_to_database(
    capture_id,
    timestamp,
    image_path,
    sensor,
    camera_metadata,
    crop_records,
    plant_activity
):

    print()
    print("========================================")
    print("Saving data to SQLite...")
    print("========================================")

    conn = sqlite3.connect(
        str(DB_FILE),
        timeout=30
    )

    try:

        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        cursor = conn.cursor()

        # ====================================================
        # INSERT CAPTURE EVENT
        # ====================================================

        cursor.execute(
            """
            INSERT INTO capture_events (

                capture_id,
                timestamp,
                full_image_path,

                water_temp_c,
                air_temp_c,
                humidity_pct,

                ph,
                ph_voltage,

                ec_ms_cm,
                tds_ppm,
                tds_voltage,

                water_level,

                camera_exposure_us,
                camera_analogue_gain,
                camera_digital_gain,
                camera_lux

            )
            VALUES (

                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?,
                ?, ?, ?, ?

            )
            """,

            (
                capture_id,

                timestamp,

                str(
                    image_path
                ),

                sensor.get(
                    "water_temp"
                ),

                sensor.get(
                    "room_temp"
                ),

                sensor.get(
                    "humidity"
                ),

                sensor.get(
                    "ph"
                ),

                sensor.get(
                    "ph_voltage"
                ),

                sensor.get(
                    "ec"
                ),

                sensor.get(
                    "tds"
                ),

                sensor.get(
                    "tds_voltage"
                ),

                sensor.get(
                    "water_level"
                ),

                camera_metadata.get(
                    "ExposureTime"
                ),

                camera_metadata.get(
                    "AnalogueGain"
                ),

                camera_metadata.get(
                    "DigitalGain"
                ),

                camera_metadata.get(
                    "Lux"
                )
            )
        )

        # ====================================================
        # INSERT P01 - P12
        # ====================================================

        active_saved = 0

        inactive_saved = 0

        for crop in crop_records:

            plant_id = (
                crop[
                    "plant_id"
                ]
            )

            plant_active = int(

                plant_activity.get(
                    plant_id,
                    0
                )
            )

            cursor.execute(
                """
                INSERT INTO plant_samples (

                    capture_id,
                    plant_id,
                    crop_path,
                    model_image_path,
                    plant_active

                )
                VALUES (

                    ?, ?, ?, ?, ?

                )
                """,

                (
                    capture_id,

                    plant_id,

                    crop[
                        "crop_path"
                    ],

                    crop[
                        "model_image_path"
                    ],

                    plant_active
                )
            )

            if plant_active == 1:

                active_saved += 1

            else:

                inactive_saved += 1

        # ----------------------------------------------------
        # COMMIT ENTIRE CAPTURE AS ONE TRANSACTION
        # ----------------------------------------------------

        conn.commit()

        print()
        print(
            "SQLite insert successful."
        )

        print(
            f"Capture ID      : "
            f"{capture_id}"
        )

        print(
            "Capture records : 1"
        )

        print(
            f"Plant records   : "
            f"{len(crop_records)}"
        )

        print(
            f"Active plants   : "
            f"{active_saved}"
        )

        print(
            f"Inactive plants : "
            f"{inactive_saved}"
        )

        print(
            f"Database        : "
            f"{DB_FILE}"
        )

    except sqlite3.Error as error:

        conn.rollback()

        print()
        print(
            "ERROR: SQLite "
            "transaction failed."
        )

        print(
            error
        )

        raise

    finally:

        conn.close()


# ============================================================
# SAVE CSV BACKUP / RESEARCH DATASET
# ============================================================

def save_dataset(
    capture_id,
    timestamp,
    image_path,
    sensor,
    camera_metadata,
    crop_records
):

    print()
    print("========================================")
    print("Saving CSV dataset...")
    print("========================================")

    # --------------------------------------------------------
    # IMPORTANT
    #
    # plant_active is intentionally NOT added here yet.
    #
    # Your existing CSV already uses this schema and changing
    # the header now would cause another old/new row mismatch.
    #
    # SQLite stores plant_active.
    # --------------------------------------------------------

    fields = [

        "capture_id",
        "timestamp",
        "plant_id",

        "full_image_path",
        "crop_path",
        "model_image_path",

        "water_temp_c",
        "air_temp_c",
        "humidity_pct",

        "ph",
        "ph_voltage",

        "ec_ms_cm",
        "tds_ppm",
        "tds_voltage",

        "water_level",

        "camera_exposure_us",
        "camera_analogue_gain",
        "camera_digital_gain",
        "camera_lux"
    ]

    file_exists = (
        CSV_FILE.exists()
    )

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fields
        )

        if not file_exists:

            writer.writeheader()

        for crop in crop_records:

            row = {

                "capture_id":
                    capture_id,

                "timestamp":
                    timestamp,

                "plant_id":
                    crop[
                        "plant_id"
                    ],

                "full_image_path":
                    str(
                        image_path
                    ),

                "crop_path":
                    crop[
                        "crop_path"
                    ],

                "model_image_path":
                    crop[
                        "model_image_path"
                    ],

                "water_temp_c":
                    sensor.get(
                        "water_temp"
                    ),

                "air_temp_c":
                    sensor.get(
                        "room_temp"
                    ),

                "humidity_pct":
                    sensor.get(
                        "humidity"
                    ),

                "ph":
                    sensor.get(
                        "ph"
                    ),

                "ph_voltage":
                    sensor.get(
                        "ph_voltage"
                    ),

                "ec_ms_cm":
                    sensor.get(
                        "ec"
                    ),

                "tds_ppm":
                    sensor.get(
                        "tds"
                    ),

                "tds_voltage":
                    sensor.get(
                        "tds_voltage"
                    ),

                "water_level":
                    sensor.get(
                        "water_level"
                    ),

                "camera_exposure_us":
                    camera_metadata.get(
                        "ExposureTime"
                    ),

                "camera_analogue_gain":
                    camera_metadata.get(
                        "AnalogueGain"
                    ),

                "camera_digital_gain":
                    camera_metadata.get(
                        "DigitalGain"
                    ),

                "camera_lux":
                    camera_metadata.get(
                        "Lux"
                    )
            }

            writer.writerow(
                row
            )

    print()

    print(
        f"{len(crop_records)} "
        f"CSV rows saved."
    )

    print(
        f"CSV: "
        f"{CSV_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print(" HYDROPONIC MULTIMODAL DATA COLLECTOR")
    print("========================================")

    # --------------------------------------------------------
    # VERIFY DATABASE
    # --------------------------------------------------------

    check_database()

    print()
    print(
        f"SQLite database ready: "
        f"{DB_FILE}"
    )

    # --------------------------------------------------------
    # GET CURRENT PLANT CONFIGURATION
    # --------------------------------------------------------

    plant_activity = (
        get_plant_activity()
    )

    display_plant_activity(
        plant_activity
    )

    # --------------------------------------------------------
    # ONE TIMESTAMP FOR ENTIRE COLLECTION EVENT
    # --------------------------------------------------------

    now = datetime.now()

    timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    timestamp_file = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    capture_id = (
        f"CAP_{timestamp_file}"
    )

    print()
    print(
        f"Capture ID   : "
        f"{capture_id}"
    )

    print(
        f"Capture time : "
        f"{timestamp}"
    )

    # --------------------------------------------------------
    # SENSOR SNAPSHOT
    # --------------------------------------------------------

    sensor = (
        read_sensor_record()
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    (
        image_path,
        camera_metadata

    ) = capture_image(
        timestamp_file
    )

    # --------------------------------------------------------
    # CROP P01-P12
    # --------------------------------------------------------

    crop_records = crop_plants(
        image_path,
        timestamp_file
    )

    if len(
        crop_records
    ) == 0:

        raise RuntimeError(
            "No plant crops were "
            "created. Nothing will "
            "be saved."
        )

    # --------------------------------------------------------
    # SQLITE
    # --------------------------------------------------------

    save_to_database(

        capture_id,

        timestamp,

        image_path,

        sensor,

        camera_metadata,

        crop_records,

        plant_activity
    )

    # --------------------------------------------------------
    # CSV BACKUP
    # --------------------------------------------------------

    save_dataset(

        capture_id,

        timestamp,

        image_path,

        sensor,

        camera_metadata,

        crop_records
    )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    active_count = sum(

        1

        for crop in crop_records

        if plant_activity.get(
            crop["plant_id"],
            0
        ) == 1
    )

    print()
    print("========================================")
    print(" COLLECTION COMPLETED SUCCESSFULLY")
    print("========================================")

    print()

    print(
        f"Capture ID : "
        f"{capture_id}"
    )

    print(
        f"Timestamp  : "
        f"{timestamp}"
    )

    print(
        f"Positions  : "
        f"{len(crop_records)}"
    )

    print(
        f"Plants     : "
        f"{active_count}"
    )

    print(
        f"Image      : "
        f"{image_path}"
    )

    print(
        f"Database   : "
        f"{DB_FILE}"
    )

    print(
        f"CSV        : "
        f"{CSV_FILE}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("========================================")
        print(" COLLECTION FAILED")
        print("========================================")

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        sys.exit(1)
