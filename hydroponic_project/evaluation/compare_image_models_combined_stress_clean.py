#!/usr/bin/env python3

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


DB_FILE = Path("/hydroponic_project/database/hydroponic.db")

MODEL_DIR = Path(
    "/hydroponic_project/models/image/candidates"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "image_model_benchmark_combined_stress_clean.csv"
)

MODEL_PATHS = {
    "MobileNetV2":
        MODEL_DIR
        / "mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth",

    "ResNet18":
        MODEL_DIR
        / "resnet18_best_healthy_unhealthy_combined_stress_clean.pth",

    "EfficientNet-B0":
        MODEL_DIR
        / "efficientnet_b0_best_healthy_unhealthy_combined_stress_clean.pth",
}

DEFAULT_SAMPLES = 50
DEFAULT_ROUNDS = 3
DEFAULT_WARMUP = 5

PYTORCH_THREADS = 4

IMAGE_START_HOUR = 6
IMAGE_END_HOUR = 16
MIN_IMAGE_LUX = 10.0
IMAGE_SIZE = 224


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class Model2Class(nn.Module):

    def __init__(
        self,
        model_name,
        num_classes=2,
    ):
        super().__init__()

        if model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(
                weights=None
            )

            in_features = (
                self.model.classifier[-1].in_features
            )

            self.model.classifier[-1] = nn.Linear(
                in_features,
                num_classes,
            )

        elif model_name == "resnet18":
            self.model = models.resnet18(
                weights=None
            )

            in_features = self.model.fc.in_features

            self.model.fc = nn.Linear(
                in_features,
                num_classes,
            )

        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(
                weights=None
            )

            in_features = (
                self.model.classifier[-1].in_features
            )

            self.model.classifier[-1] = nn.Linear(
                in_features,
                num_classes,
            )

        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

    def forward(self, x):
        return self.model(x)


def load_candidate_images(
    requested_samples,
):
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_FILE}"
        )

    conn = sqlite3.connect(str(DB_FILE))

    try:
        df = pd.read_sql_query(
            '''
            SELECT DISTINCT
                ce.timestamp,
                ce.camera_lux,
                ps.capture_id,
                ps.plant_id,
                ps.model_image_path
            FROM plant_samples ps
            INNER JOIN capture_events ce
                ON ce.capture_id = ps.capture_id
            WHERE
                ps.plant_active = 1
                AND LOWER(
                    TRIM(
                        COALESCE(
                            ps.image_prediction,
                            ''
                        )
                    )
                ) IN (
                    'healthy',
                    'unhealthy'
                )
                AND CAST(
                    strftime(
                        '%H',
                        ce.timestamp
                    )
                    AS INTEGER
                )
                BETWEEN ?
                AND ?
                AND ce.camera_lux >= ?
            ORDER BY
                ce.timestamp,
                ps.plant_id
            ''',
            conn,
            params=[
                IMAGE_START_HOUR,
                IMAGE_END_HOUR,
                MIN_IMAGE_LUX,
            ],
        )

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "No eligible daytime/lux-qualified images "
            "were found in SQLite."
        )

    existing_rows = []

    for _, row in df.iterrows():
        path = Path(
            str(row["model_image_path"])
        )

        if path.exists():
            existing_rows.append(
                {
                    "timestamp":
                        row["timestamp"],
                    "camera_lux":
                        row["camera_lux"],
                    "capture_id":
                        row["capture_id"],
                    "plant_id":
                        row["plant_id"],
                    "image_path":
                        str(path),
                }
            )

    existing_df = pd.DataFrame(
        existing_rows
    )

    if existing_df.empty:
        raise RuntimeError(
            "Eligible database rows were found, "
            "but none of their image files exist."
        )

    existing_df = (
        existing_df
        .drop_duplicates(
            subset=["image_path"]
        )
        .reset_index(drop=True)
    )

    if len(existing_df) > requested_samples:
        sample_indexes = np.linspace(
            0,
            len(existing_df) - 1,
            num=requested_samples,
            dtype=int,
        )

        existing_df = (
            existing_df
            .iloc[sample_indexes]
            .reset_index(drop=True)
        )

    return existing_df


def preprocess_image(
    image_path,
):
    with Image.open(
        image_path
    ) as image:
        image = image.convert("RGB")
        tensor = IMAGE_TRANSFORM(image)

    return tensor.unsqueeze(0)


def load_model(
    checkpoint_path,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    start = time.perf_counter()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_name = checkpoint.get(
        "model_name"
    )

    if model_name is None:
        raise RuntimeError(
            f"Checkpoint missing model_name: "
            f"{checkpoint_path}"
        )

    class_names = checkpoint.get(
        "class_names",
        ["healthy", "unhealthy"],
    )

    class_mapping = checkpoint.get(
        "class_to_idx",
        {
            "healthy": 0,
            "unhealthy": 1,
        },
    )

    expected_mapping = {
        "healthy": 0,
        "unhealthy": 1,
    }

    if class_mapping != expected_mapping:
        raise RuntimeError(
            f"Unexpected class mapping in "
            f"{checkpoint_path.name}: "
            f"{class_mapping}"
        )

    model = Model2Class(
        model_name=model_name,
        num_classes=len(class_names),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    load_ms = (
        time.perf_counter() - start
    ) * 1000.0

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    return {
        "model":
            model,
        "model_name":
            model_name,
        "class_names":
            class_names,
        "class_mapping":
            class_mapping,
        "best_epoch":
            checkpoint.get("best_epoch"),
        "best_val_macro_f1":
            checkpoint.get(
                "best_val_macro_f1",
                checkpoint.get("best_val_f1"),
            ),
        "load_ms":
            load_ms,
        "total_parameters":
            total_parameters,
    }


def benchmark_model(
    display_name,
    checkpoint_path,
    image_df,
    rounds,
    warmup,
):
    print()
    print("=" * 64)
    print(
        f"Benchmarking: {display_name}"
    )
    print("=" * 64)

    metadata = load_model(
        checkpoint_path
    )

    model = metadata["model"]

    preprocessed_tensors = [
        preprocess_image(path)
        for path in image_df["image_path"]
    ]

    with torch.inference_mode():
        for warmup_index in range(warmup):
            tensor = preprocessed_tensors[
                warmup_index
                % len(preprocessed_tensors)
            ]
            _ = model(tensor)

    inference_times_ms = []

    with torch.inference_mode():
        for _ in range(rounds):
            for tensor in preprocessed_tensors:
                start = time.perf_counter()
                _ = model(tensor)
                inference_times_ms.append(
                    (
                        time.perf_counter()
                        - start
                    ) * 1000.0
                )

    end_to_end_times_ms = []

    with torch.inference_mode():
        for _ in range(rounds):
            for image_path in image_df[
                "image_path"
            ]:
                start = time.perf_counter()
                tensor = preprocess_image(
                    image_path
                )
                _ = model(tensor)
                end_to_end_times_ms.append(
                    (
                        time.perf_counter()
                        - start
                    ) * 1000.0
                )

    inference_array = np.asarray(
        inference_times_ms,
        dtype=np.float64,
    )

    end_to_end_array = np.asarray(
        end_to_end_times_ms,
        dtype=np.float64,
    )

    mean_inference_ms = float(
        np.mean(inference_array)
    )

    median_inference_ms = float(
        np.median(inference_array)
    )

    p95_inference_ms = float(
        np.percentile(
            inference_array,
            95,
        )
    )

    mean_end_to_end_ms = float(
        np.mean(end_to_end_array)
    )

    throughput_img_s = (
        1000.0
        / mean_inference_ms
    )

    file_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    parameter_million = (
        metadata["total_parameters"]
        / 1_000_000
    )

    print(
        f"File size           : "
        f"{file_size_mb:.3f} MB"
    )

    print(
        f"Parameters          : "
        f"{parameter_million:.3f} M"
    )

    print(
        f"Best epoch          : "
        f"{metadata['best_epoch']}"
    )

    if metadata[
        "best_val_macro_f1"
    ] is not None:
        print(
            f"Best val Macro-F1   : "
            f"{metadata['best_val_macro_f1']:.6f}"
        )

    print(
        f"Load time           : "
        f"{metadata['load_ms']:.2f} ms"
    )

    print(
        f"Mean inference      : "
        f"{mean_inference_ms:.2f} ms"
    )

    print(
        f"Median inference    : "
        f"{median_inference_ms:.2f} ms"
    )

    print(
        f"P95 inference       : "
        f"{p95_inference_ms:.2f} ms"
    )

    print(
        f"Mean end-to-end     : "
        f"{mean_end_to_end_ms:.2f} ms"
    )

    print(
        f"Throughput          : "
        f"{throughput_img_s:.2f} images/s"
    )

    print(
        f"Unique images       : "
        f"{len(image_df)}"
    )

    print(
        f"Benchmark rounds    : "
        f"{rounds}"
    )

    print(
        f"Timed inferences    : "
        f"{len(inference_times_ms)}"
    )

    result = {
        "model":
            display_name,
        "checkpoint":
            str(checkpoint_path),
        "architecture":
            metadata["model_name"],
        "file_size_mb":
            file_size_mb,
        "parameters":
            metadata["total_parameters"],
        "parameters_m":
            parameter_million,
        "best_epoch":
            metadata["best_epoch"],
        "best_val_macro_f1":
            metadata["best_val_macro_f1"],
        "load_ms":
            metadata["load_ms"],
        "mean_inference_ms":
            mean_inference_ms,
        "median_inference_ms":
            median_inference_ms,
        "p95_inference_ms":
            p95_inference_ms,
        "mean_end_to_end_ms":
            mean_end_to_end_ms,
        "throughput_img_s":
            throughput_img_s,
        "unique_images":
            len(image_df),
        "rounds":
            rounds,
        "timed_inferences":
            len(inference_times_ms),
    }

    del model
    del preprocessed_tensors

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the new clean-stress "
            "CNN checkpoints on Raspberry Pi CPU."
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    args = parser.parse_args()

    if args.samples < 1:
        raise ValueError(
            "--samples must be >= 1"
        )

    if args.rounds < 1:
        raise ValueError(
            "--rounds must be >= 1"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be >= 0"
        )

    torch.set_num_threads(
        PYTORCH_THREADS
    )

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for display_name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{display_name} checkpoint "
                f"missing: {path}"
            )

    image_df = load_candidate_images(
        args.samples
    )

    print()
    print("=" * 64)
    print(
        " RASPBERRY PI IMAGE MODEL EDGE BENCHMARK"
    )
    print("=" * 64)

    print("Device             : cpu")
    print(
        f"PyTorch threads    : "
        f"{torch.get_num_threads()}"
    )
    print(
        f"Image time gate    : "
        f"{IMAGE_START_HOUR:02d}:00 - "
        f"{IMAGE_END_HOUR:02d}:59"
    )
    print(
        f"Minimum Lux        : "
        f"{MIN_IMAGE_LUX}"
    )
    print(
        f"Requested samples  : "
        f"{args.samples}"
    )
    print(
        f"Benchmark rounds   : "
        f"{args.rounds}"
    )
    print(
        f"Warmup inferences  : "
        f"{args.warmup}"
    )
    print(
        f"Usable images      : "
        f"{len(image_df)}"
    )

    results = []

    model_items = list(
        MODEL_PATHS.items()
    )

    for index, (
        display_name,
        checkpoint_path,
    ) in enumerate(model_items):

        result = benchmark_model(
            display_name,
            checkpoint_path,
            image_df,
            args.rounds,
            args.warmup,
        )

        results.append(result)

        if index < len(model_items) - 1:
            print(
                "Cooling down for 2.0 seconds..."
            )
            time.sleep(2.0)

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "mean_inference_ms",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 64)
    print(" BENCHMARK SUMMARY")
    print("=" * 64)

    summary_columns = [
        "model",
        "file_size_mb",
        "parameters_m",
        "load_ms",
        "mean_inference_ms",
        "median_inference_ms",
        "p95_inference_ms",
        "mean_end_to_end_ms",
        "throughput_img_s",
        "best_epoch",
        "best_val_macro_f1",
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )

    fastest = results_df.iloc[0]

    smallest = (
        results_df
        .sort_values(
            "file_size_mb"
        )
        .iloc[0]
    )

    print()
    print(
        "Fastest on this Raspberry Pi : "
        f"{fastest['model']} "
        f"({fastest['mean_inference_ms']:.2f} ms/image)"
    )

    print(
        "Smallest model file          : "
        f"{smallest['model']} "
        f"({smallest['file_size_mb']:.2f} MB)"
    )

    print()
    print(
        "IMPORTANT: This benchmark measures edge "
        "efficiency only, not accuracy."
    )

    print(
        "Next, evaluate these same checkpoints against "
        "the manually labelled live plant observations."
    )

    print()
    print(
        "CSV saved to:"
    )
    print(
        OUTPUT_CSV
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
