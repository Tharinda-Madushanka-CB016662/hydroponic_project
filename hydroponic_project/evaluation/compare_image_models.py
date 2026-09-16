import argparse
import gc
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

MODEL_DIR = Path(
    "/hydroponic_project/models/image"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation"
)

OUTPUT_CSV = OUTPUT_DIR / "image_model_benchmark.csv"


# ============================================================
# MODELS TO COMPARE
# ============================================================

MODEL_FILES = {
    "MobileNetV2": MODEL_DIR / "mobilenet_v2_best_healthy_unhealthy.pth",
    "ResNet18": MODEL_DIR / "resnet18_best_healthy_unhealthy.pth",
    "EfficientNet-B0": MODEL_DIR / "efficientnet_b0_best_healthy_unhealthy.pth",
}


# ============================================================
# DEVICE / PREPROCESSING
# ============================================================

DEVICE = torch.device("cpu")
torch.set_num_threads(4)

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# GENERIC WRAPPER
# ============================================================
#
# Your saved checkpoints use a wrapper with the underlying
# torchvision model stored as self.model. This class recreates
# that structure for all three architectures.
# ============================================================

class EdgeImageClassifier(nn.Module):
    def __init__(
        self,
        architecture,
        num_classes=2,
    ):
        super().__init__()

        architecture = architecture.lower()

        if architecture in {
            "mobilenet_v2",
            "mobilenetv2",
            "mobilenet",
        }:
            self.model = models.mobilenet_v2(
                weights=None
            )

            in_features = self.model.classifier[1].in_features

            self.model.classifier[1] = nn.Linear(
                in_features,
                num_classes,
            )

        elif architecture in {
            "resnet18",
            "resnet_18",
            "resnet",
        }:
            self.model = models.resnet18(
                weights=None
            )

            in_features = self.model.fc.in_features

            self.model.fc = nn.Linear(
                in_features,
                num_classes,
            )

        elif architecture in {
            "efficientnet_b0",
            "efficientnet-b0",
            "efficientnetb0",
            "efficientnet",
        }:
            self.model = models.efficientnet_b0(
                weights=None
            )

            in_features = self.model.classifier[1].in_features

            self.model.classifier[1] = nn.Linear(
                in_features,
                num_classes,
            )

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}"
            )

    def forward(self, x):
        return self.model(x)


# ============================================================
# SETTINGS
# ============================================================

def get_quality_settings():
    """
    Use the same quality gate as production when selecting
    benchmark images. Falls back to 06:00-16:59 and Lux >= 10.
    """

    defaults = {
        "min_image_lux": 10.0,
        "image_start_hour": 6,
        "image_end_hour": 17,
    }

    if not DB_FILE.exists():
        return defaults

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    try:
        table = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name='system_settings'
            """
        ).fetchone()

        if table is None:
            return defaults

        rows = conn.execute(
            """
            SELECT setting_key, setting_value
            FROM system_settings
            WHERE setting_key IN (
                'min_image_lux',
                'image_start_hour',
                'image_end_hour'
            )
            """
        ).fetchall()

        raw = {
            key: value
            for key, value in rows
        }

        return {
            "min_image_lux": float(
                raw.get(
                    "min_image_lux",
                    defaults["min_image_lux"],
                )
            ),
            "image_start_hour": int(
                float(
                    raw.get(
                        "image_start_hour",
                        defaults["image_start_hour"],
                    )
                )
            ),
            "image_end_hour": int(
                float(
                    raw.get(
                        "image_end_hour",
                        defaults["image_end_hour"],
                    )
                )
            ),
        }

    finally:
        conn.close()


# ============================================================
# SELECT REAL RASPBERRY PI IMAGES
# ============================================================

def get_benchmark_images(
    requested_samples,
    settings,
):
    """
    Select recent active plant crops that passed the production
    time/Lux gate. The same image set is used for every model.
    """

    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}"
        )

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    try:
        query = """
        SELECT
            ps.model_image_path,
            ps.capture_id,
            ps.plant_id,
            ce.timestamp,
            ce.camera_lux
        FROM plant_samples ps
        JOIN capture_events ce
            ON ce.capture_id = ps.capture_id
        WHERE ps.plant_active = 1
          AND CAST(strftime('%H', ce.timestamp) AS INTEGER) >= ?
          AND CAST(strftime('%H', ce.timestamp) AS INTEGER) < ?
          AND ce.camera_lux IS NOT NULL
          AND ce.camera_lux >= ?
        ORDER BY ce.timestamp DESC, ps.plant_id
        LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                settings["image_start_hour"],
                settings["image_end_hour"],
                settings["min_image_lux"],
                requested_samples * 3,
            ),
        )

    finally:
        conn.close()

    # SQLite stores paths; validate that files still exist.
    valid_rows = []

    for _, row in df.iterrows():
        path = Path(
            row["model_image_path"]
        )

        if path.exists():
            valid_rows.append(
                row.to_dict()
            )

        if len(valid_rows) >= requested_samples:
            break

    if not valid_rows:
        raise RuntimeError(
            "No valid active plant images passed the current "
            "time/Lux quality gate."
        )

    return valid_rows


# ============================================================
# CHECKPOINT ARCHITECTURE DETECTION
# ============================================================

def infer_architecture(
    checkpoint,
    filename,
):
    name = str(
        checkpoint.get(
            "model_name",
            "",
        )
    ).lower()

    if "mobile" in name:
        return "mobilenet_v2"

    if "resnet" in name:
        return "resnet18"

    if "efficient" in name:
        return "efficientnet_b0"

    lower_filename = filename.lower()

    if "mobilenet" in lower_filename:
        return "mobilenet_v2"

    if "resnet" in lower_filename:
        return "resnet18"

    if "efficientnet" in lower_filename:
        return "efficientnet_b0"

    raise RuntimeError(
        f"Unable to determine architecture for {filename}"
    )


# ============================================================
# MODEL LOAD
# ============================================================

def load_model(model_path):
    start = time.perf_counter()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    architecture = infer_architecture(
        checkpoint,
        model_path.name,
    )

    num_classes = int(
        checkpoint.get(
            "num_classes",
            2,
        )
    )

    model = EdgeImageClassifier(
        architecture,
        num_classes=num_classes,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    # First try the expected wrapper-prefixed state dict.
    try:
        model.load_state_dict(
            state_dict
        )

    except RuntimeError:
        # Fallback for checkpoints saved directly from the
        # underlying torchvision network without "model." prefix.
        model.model.load_state_dict(
            state_dict
        )

    model.to(DEVICE)
    model.eval()

    load_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        model,
        checkpoint,
        architecture,
        load_ms,
    )


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(
    display_name,
    model_path,
    image_rows,
    warmup_runs,
    rounds,
):
    if not model_path.exists():
        return {
            "model": display_name,
            "status": "MODEL FILE MISSING",
            "model_file": str(model_path),
        }

    print()
    print("============================================================")
    print(f"Benchmarking: {display_name}")
    print("============================================================")

    (
        model,
        checkpoint,
        architecture,
        load_ms,
    ) = load_model(
        model_path
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    file_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    # Warm up the CPU kernels before measuring latency.
    warmup_image = Image.open(
        image_rows[0]["model_image_path"]
    ).convert("RGB")

    warmup_tensor = IMAGE_TRANSFORM(
        warmup_image
    ).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(
                warmup_tensor
            )

    inference_times = []
    end_to_end_times = []
    predictions = []

    class_names = checkpoint.get(
        "class_names",
        ["healthy", "unhealthy"],
    )

    for _round in range(rounds):
        for row in image_rows:
            image_path = row[
                "model_image_path"
            ]

            end_to_end_start = time.perf_counter()

            image = Image.open(
                image_path
            ).convert("RGB")

            tensor = IMAGE_TRANSFORM(
                image
            ).unsqueeze(0).to(DEVICE)

            inference_start = time.perf_counter()

            with torch.inference_mode():
                logits = model(
                    tensor
                )

            inference_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000

            end_to_end_ms = (
                time.perf_counter()
                - end_to_end_start
            ) * 1000

            prediction_index = int(
                logits.argmax(
                    dim=1
                ).item()
            )

            predictions.append(
                class_names[
                    prediction_index
                ]
                if prediction_index < len(class_names)
                else str(prediction_index)
            )

            inference_times.append(
                inference_ms
            )

            end_to_end_times.append(
                end_to_end_ms
            )

    inference_array = np.asarray(
        inference_times,
        dtype=float,
    )

    e2e_array = np.asarray(
        end_to_end_times,
        dtype=float,
    )

    mean_inference_ms = float(
        inference_array.mean()
    )

    result = {
        "model": display_name,
        "architecture": architecture,
        "status": "SUCCESS",
        "model_file": str(model_path),
        "file_size_mb": round(file_size_mb, 3),
        "parameters_m": round(parameter_count / 1_000_000, 3),
        "load_ms": round(load_ms, 3),
        "mean_inference_ms": round(mean_inference_ms, 3),
        "median_inference_ms": round(
            float(np.median(inference_array)),
            3,
        ),
        "p95_inference_ms": round(
            float(np.percentile(inference_array, 95)),
            3,
        ),
        "mean_end_to_end_ms": round(
            float(e2e_array.mean()),
            3,
        ),
        "throughput_img_s": round(
            1000.0 / mean_inference_ms,
            3,
        ) if mean_inference_ms > 0 else None,
        "samples": len(image_rows),
        "rounds": rounds,
        "timed_inferences": len(inference_times),
        "best_val_f1": checkpoint.get("best_val_f1"),
        "best_epoch": checkpoint.get("best_epoch"),
        "healthy_predictions": predictions.count("healthy"),
        "unhealthy_predictions": predictions.count("unhealthy"),
    }

    print(f"File size           : {result['file_size_mb']:.3f} MB")
    print(f"Parameters          : {result['parameters_m']:.3f} M")
    print(f"Load time           : {result['load_ms']:.2f} ms")
    print(f"Mean inference      : {result['mean_inference_ms']:.2f} ms")
    print(f"Median inference    : {result['median_inference_ms']:.2f} ms")
    print(f"P95 inference       : {result['p95_inference_ms']:.2f} ms")
    print(f"Mean end-to-end     : {result['mean_end_to_end_ms']:.2f} ms")
    print(f"Throughput          : {result['throughput_img_s']:.2f} images/s")
    print(f"Unique images       : {result['samples']}")
    print(f"Benchmark rounds    : {result['rounds']}")
    print(f"Timed inferences    : {result['timed_inferences']}")

    del model
    del checkpoint
    gc.collect()

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of real plant crops used per model. Default: 50",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warm-up inference runs before timing. Default: 5",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Repeat the same image set this many times. Default: 3",
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Seconds to wait between models to reduce thermal bias. Default: 2",
    )

    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError(
            "--samples must be at least 1"
        )

    if args.rounds < 1:
        raise ValueError(
            "--rounds must be at least 1"
        )

    settings = get_quality_settings()

    print()
    print("============================================================")
    print(" RASPBERRY PI IMAGE MODEL EDGE BENCHMARK")
    print("============================================================")
    print(f"Device             : {DEVICE}")
    print(f"PyTorch threads    : {torch.get_num_threads()}")
    print(
        f"Image time gate    : "
        f"{settings['image_start_hour']:02d}:00 - "
        f"{settings['image_end_hour'] - 1:02d}:59"
    )
    print(f"Minimum Lux        : {settings['min_image_lux']}")
    print(f"Requested samples  : {args.samples}")
    print(f"Benchmark rounds   : {args.rounds}")

    image_rows = get_benchmark_images(
        args.samples,
        settings,
    )

    print(f"Usable images      : {len(image_rows)}")

    results = []

    for display_name, model_path in MODEL_FILES.items():
        try:
            result = benchmark_model(
                display_name,
                model_path,
                image_rows,
                args.warmup,
                args.rounds,
            )

        except Exception as error:
            result = {
                "model": display_name,
                "status": "FAILED",
                "model_file": str(model_path),
                "error": f"{type(error).__name__}: {error}",
            }

            print()
            print(f"{display_name} FAILED: {result['error']}")

        results.append(
            result
        )

        if args.cooldown > 0:
            print(
                f"Cooling down for {args.cooldown:.1f} seconds..."
            )
            time.sleep(args.cooldown)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    successful = result_df[
        result_df["status"] == "SUCCESS"
    ].copy()

    print()
    print("============================================================")
    print(" BENCHMARK SUMMARY")
    print("============================================================")

    if successful.empty:
        print("No model completed successfully.")

    else:
        successful = successful.sort_values(
            "mean_inference_ms"
        )

        columns = [
            "model",
            "file_size_mb",
            "parameters_m",
            "load_ms",
            "mean_inference_ms",
            "p95_inference_ms",
            "throughput_img_s",
            "best_val_f1",
        ]

        available_columns = [
            col
            for col in columns
            if col in successful.columns
        ]

        print(
            successful[
                available_columns
            ].to_string(
                index=False
            )
        )

        fastest_row = successful.iloc[0]

        smallest_row = successful.loc[
            successful["file_size_mb"].idxmin()
        ]

        print()
        print(
            f"Fastest on this Raspberry Pi : "
            f"{fastest_row['model']} "
            f"({fastest_row['mean_inference_ms']:.2f} ms/image)"
        )

        print(
            f"Smallest model file          : "
            f"{smallest_row['model']} "
            f"({smallest_row['file_size_mb']:.2f} MB)"
        )

    print()
    print(f"CSV saved to: {OUTPUT_CSV}")
    print()
    print(
        "IMPORTANT: latency/size measure edge deployment cost. "
        "Use held-out test F1/accuracy from your training evaluation "
        "alongside this benchmark before selecting the final model."
    )
    print()


if __name__ == "__main__":
    main()
