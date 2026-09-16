#!/usr/bin/env python3

# ============================================================
# LIVE GROUND-TRUTH EVALUATION FOR NEW IMAGE MODELS
# ============================================================
#
# Purpose
# -------
# Evaluate the NEW stress-trained image models against the
# manually labelled Raspberry Pi observations already stored
# in SQLite.
#
# This script is intentionally OFFLINE / READ-ONLY:
#
#   - it DOES NOT update plant_samples
#   - it DOES NOT change production inference
#   - it DOES NOT change fusion predictions
#   - it DOES NOT change Grad-CAM paths
#
# It only reads:
#
#   manual_labels
#   plant_samples
#   capture_events
#
# and runs candidate CNN inference on the corresponding
# model_image_path files.
#
# Models evaluated
# ----------------
#   1. Existing production MobileNetV2 (baseline), if present
#   2. New MobileNetV2
#   3. New ResNet18
#   4. New EfficientNet-B0
#
# Ground-truth labels used
# ------------------------
#   healthy
#   unhealthy
#
# Samples labelled "uncertain" are excluded from binary
# evaluation.
#
# Outputs
# -------
# /hydroponic_project/evaluation/live_model_evaluation/
#
#   live_image_model_predictions.csv
#   live_image_model_metrics.csv
#   live_image_model_confusion_matrices.csv
#
# ============================================================

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image

from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

PRODUCTION_MODEL = Path(
    "/hydroponic_project/models/image/"
    "mobilenet_v2_best_healthy_unhealthy.pth"
)

CANDIDATE_DIR = Path(
    "/hydroponic_project/models/image/candidates"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/live_model_evaluation"
)

PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "live_image_model_predictions.csv"
)

METRICS_CSV = (
    OUTPUT_DIR
    / "live_image_model_metrics.csv"
)

CONFUSION_CSV = (
    OUTPUT_DIR
    / "live_image_model_confusion_matrices.csv"
)


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATHS = {
    "Old_MobileNetV2":
        PRODUCTION_MODEL,

    "New_MobileNetV2":
        CANDIDATE_DIR
        / "mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth",

    "New_ResNet18":
        CANDIDATE_DIR
        / "resnet18_best_healthy_unhealthy_combined_stress_clean.pth",

    "New_EfficientNet_B0":
        CANDIDATE_DIR
        / "efficientnet_b0_best_healthy_unhealthy_combined_stress_clean.pth",
}


# ============================================================
# INFERENCE SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

PYTORCH_THREADS = 4

IMAGE_SIZE = 224

CLASS_NAMES = [
    "healthy",
    "unhealthy",
]

EXPECTED_MAPPING = {
    "healthy": 0,
    "unhealthy": 1,
}


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL WRAPPER
# ============================================================

class Model2Class(
    nn.Module
):

    def __init__(
        self,
        model_name,
        num_classes=2,
    ):

        super().__init__()

        if model_name == "mobilenet_v2":

            self.model = (
                models.mobilenet_v2(
                    weights=None
                )
            )

            in_features = (
                self.model
                .classifier[-1]
                .in_features
            )

            self.model.classifier[-1] = (
                nn.Linear(
                    in_features,
                    num_classes,
                )
            )


        elif model_name == "resnet18":

            self.model = (
                models.resnet18(
                    weights=None
                )
            )

            in_features = (
                self.model.fc.in_features
            )

            self.model.fc = (
                nn.Linear(
                    in_features,
                    num_classes,
                )
            )


        elif model_name == "efficientnet_b0":

            self.model = (
                models.efficientnet_b0(
                    weights=None
                )
            )

            in_features = (
                self.model
                .classifier[-1]
                .in_features
            )

            self.model.classifier[-1] = (
                nn.Linear(
                    in_features,
                    num_classes,
                )
            )


        else:

            raise ValueError(
                f"Unsupported model architecture: "
                f"{model_name}"
            )


    def forward(
        self,
        x,
    ):

        return self.model(
            x
        )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def infer_architecture_from_filename(
    checkpoint_path,
):
    """
    Fallback for an older checkpoint that may not contain a
    model_name metadata field.
    """

    name = (
        checkpoint_path
        .name
        .lower()
    )

    if "mobilenet" in name:
        return "mobilenet_v2"

    if "resnet18" in name:
        return "resnet18"

    if (
        "efficientnet"
        in name
    ):
        return "efficientnet_b0"

    raise RuntimeError(
        f"Could not infer architecture from "
        f"{checkpoint_path.name}"
    )


def load_checkpoint_model(
    checkpoint_path,
):
    """
    Load one CNN checkpoint safely on CPU.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    model_name = checkpoint.get(
        "model_name"
    )


    if model_name is None:

        model_name = (
            infer_architecture_from_filename(
                checkpoint_path
            )
        )


    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )


    class_mapping = checkpoint.get(
        "class_to_idx",
        EXPECTED_MAPPING,
    )


    if class_mapping != EXPECTED_MAPPING:

        raise RuntimeError(
            f"Unexpected class mapping in "
            f"{checkpoint_path.name}: "
            f"{class_mapping}"
        )


    model = Model2Class(
        model_name=
            model_name,

        num_classes=
            len(
                class_names
            ),
    ).to(
        DEVICE
    )


    # Older checkpoints may have either:
    #
    #   model_state_dict
    #
    # or be a direct state_dict.
    #
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = (
            checkpoint
        )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    return {
        "model":
            model,

        "model_name":
            model_name,

        "class_names":
            class_names,

        "best_epoch":
            (
                checkpoint.get(
                    "best_epoch"
                )
                if isinstance(
                    checkpoint,
                    dict,
                )
                else None
            ),
    }


# ============================================================
# LOAD MANUALLY LABELLED OBSERVATIONS
# ============================================================

def load_ground_truth():
    """
    Load only binary Healthy / Unhealthy manual labels.

    'uncertain' labels are intentionally excluded.
    """

    if not DB_FILE.exists():

        raise FileNotFoundError(
            f"Database not found: "
            f"{DB_FILE}"
        )


    conn = sqlite3.connect(
        str(
            DB_FILE
        )
    )


    try:

        df = pd.read_sql_query(
            """
            SELECT
                ce.timestamp,
                ce.capture_id,
                ce.camera_lux,

                ps.plant_id,
                ps.plant_active,
                ps.model_image_path,

                ps.image_prediction
                    AS old_stored_image_prediction,

                ps.image_confidence
                    AS old_stored_image_confidence,

                ml.ground_truth_label,
                ml.validation_method,
                ml.reviewer_role,
                ml.notes

            FROM manual_labels ml

            INNER JOIN plant_samples ps
                ON ps.capture_id = ml.capture_id
               AND ps.plant_id = ml.plant_id

            INNER JOIN capture_events ce
                ON ce.capture_id = ml.capture_id

            WHERE
                ps.plant_active = 1

                AND LOWER(
                    TRIM(
                        ml.ground_truth_label
                    )
                ) IN (
                    'healthy',
                    'unhealthy'
                )

            ORDER BY
                ce.timestamp,
                ps.plant_id
            """,
            conn,
        )


    finally:

        conn.close()


    if df.empty:

        raise RuntimeError(
            "No Healthy/Unhealthy manual labels "
            "were found."
        )


    df[
        "ground_truth"
    ] = (
        df[
            "ground_truth_label"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    # --------------------------------------------------------
    # KEEP ONLY OBSERVATIONS WHOSE IMAGE STILL EXISTS
    # --------------------------------------------------------

    df[
        "image_exists"
    ] = df[
        "model_image_path"
    ].apply(
        lambda path:
            Path(
                str(
                    path
                )
            ).exists()
    )


    missing_count = int(
        (
            ~df[
                "image_exists"
            ]
        ).sum()
    )


    if missing_count > 0:

        print(
            f"WARNING: {missing_count} labelled "
            "observations have missing image files "
            "and will be excluded."
        )


    df = (
        df[
            df[
                "image_exists"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if df.empty:

        raise RuntimeError(
            "No labelled observations have an "
            "existing model_image_path."
        )


    return df


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image_path,
):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        tensor = IMAGE_TRANSFORM(
            image
        )


    return tensor.unsqueeze(
        0
    )


# ============================================================
# RUN ONE MODEL ACROSS ALL LABELLED IMAGES
# ============================================================

def run_model_predictions(
    model,
    dataframe,
):
    """
    Returns:
        predicted labels
        unhealthy probabilities
        predicted-class confidences
    """

    predicted_labels = []
    unhealthy_probabilities = []
    predicted_confidences = []


    with torch.inference_mode():

        for image_path in dataframe[
            "model_image_path"
        ]:

            tensor = preprocess_image(
                image_path
            ).to(
                DEVICE
            )


            logits = model(
                tensor
            )


            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]


            unhealthy_probability = float(
                probabilities[1].item()
            )


            predicted_index = int(
                torch.argmax(
                    probabilities
                ).item()
            )


            predicted_label = (
                CLASS_NAMES[
                    predicted_index
                ]
            )


            predicted_confidence = float(
                probabilities[
                    predicted_index
                ].item()
            )


            predicted_labels.append(
                predicted_label
            )


            unhealthy_probabilities.append(
                unhealthy_probability
            )


            predicted_confidences.append(
                predicted_confidence
            )


    return (
        predicted_labels,
        unhealthy_probabilities,
        predicted_confidences,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Positive class:
        unhealthy
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "healthy",
            "unhealthy",
        ],
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "unhealthy_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
                    zero_division=0,
                )
            ),

        "unhealthy_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
                    zero_division=0,
                )
            ),

        "unhealthy_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label="unhealthy",
                    zero_division=0,
                )
            ),

        "healthy_correct_tn":
            int(
                tn
            ),

        "healthy_as_unhealthy_fp":
            int(
                fp
            ),

        "unhealthy_as_healthy_fn":
            int(
                fn
            ),

        "unhealthy_correct_tp":
            int(
                tp
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.set_num_threads(
        PYTORCH_THREADS
    )


    try:

        torch.set_num_interop_threads(
            1
        )

    except RuntimeError:

        pass


    df = load_ground_truth()


    healthy_count = int(
        (
            df[
                "ground_truth"
            ]
            == "healthy"
        ).sum()
    )


    unhealthy_count = int(
        (
            df[
                "ground_truth"
            ]
            == "unhealthy"
        ).sum()
    )


    print()
    print("=" * 92)
    print(
        " LIVE MANUAL-GROUND-TRUTH IMAGE MODEL EVALUATION"
    )
    print("=" * 92)


    print(
        f"Labelled observations : "
        f"{len(df)}"
    )


    print(
        f"Healthy              : "
        f"{healthy_count}"
    )


    print(
        f"Unhealthy            : "
        f"{unhealthy_count}"
    )


    print(
        f"Unique captures      : "
        f"{df['capture_id'].nunique()}"
    )


    print(
        f"Unique plants        : "
        f"{df['plant_id'].nunique()}"
    )


    print(
        f"Device               : "
        f"{DEVICE}"
    )


    print()


    metric_rows = []
    confusion_rows = []


    available_models = {
        name:
            path

        for name, path
        in MODEL_PATHS.items()

        if path.exists()
    }


    missing_models = {
        name:
            path

        for name, path
        in MODEL_PATHS.items()

        if not path.exists()
    }


    for name, path in (
        missing_models.items()
    ):

        print(
            f"WARNING: Skipping missing model "
            f"{name}: {path}"
        )


    if not available_models:

        raise RuntimeError(
            "None of the configured model "
            "checkpoints exist."
        )


    for (
        display_name,
        checkpoint_path,
    ) in available_models.items():


        print("=" * 92)

        print(
            f"Evaluating: "
            f"{display_name}"
        )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )


        metadata = load_checkpoint_model(
            checkpoint_path
        )


        (
            predicted_labels,
            unhealthy_probs,
            predicted_confidences,
        ) = run_model_predictions(
            metadata[
                "model"
            ],
            df,
        )


        prediction_column = (
            f"{display_name}_prediction"
        )


        unhealthy_probability_column = (
            f"{display_name}_unhealthy_probability"
        )


        confidence_column = (
            f"{display_name}_confidence"
        )


        df[
            prediction_column
        ] = predicted_labels


        df[
            unhealthy_probability_column
        ] = unhealthy_probs


        df[
            confidence_column
        ] = predicted_confidences


        metrics = calculate_metrics(
            df[
                "ground_truth"
            ],
            df[
                prediction_column
            ],
        )


        metric_row = {
            "model":
                display_name,

            "architecture":
                metadata[
                    "model_name"
                ],

            "samples":
                len(
                    df
                ),

            "healthy_ground_truth":
                healthy_count,

            "unhealthy_ground_truth":
                unhealthy_count,

            "best_epoch":
                metadata[
                    "best_epoch"
                ],
        }


        metric_row.update(
            metrics
        )


        metric_rows.append(
            metric_row
        )


        confusion_rows.append(
            {
                "model":
                    display_name,

                "healthy_to_healthy":
                    metrics[
                        "healthy_correct_tn"
                    ],

                "healthy_to_unhealthy":
                    metrics[
                        "healthy_as_unhealthy_fp"
                    ],

                "unhealthy_to_healthy":
                    metrics[
                        "unhealthy_as_healthy_fn"
                    ],

                "unhealthy_to_unhealthy":
                    metrics[
                        "unhealthy_correct_tp"
                    ],
            }
        )


        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.4f}"
        )


        print(
            f"Balanced Accuracy : "
            f"{metrics['balanced_accuracy']:.4f}"
        )


        print(
            f"Macro-F1          : "
            f"{metrics['macro_f1']:.4f}"
        )


        print(
            f"Unhealthy Precision: "
            f"{metrics['unhealthy_precision']:.4f}"
        )


        print(
            f"Unhealthy Recall   : "
            f"{metrics['unhealthy_recall']:.4f}"
        )


        print(
            f"Unhealthy F1       : "
            f"{metrics['unhealthy_f1']:.4f}"
        )


        print(
            "Confusion counts   : "
            f"TN={metrics['healthy_correct_tn']} "
            f"FP={metrics['healthy_as_unhealthy_fp']} "
            f"FN={metrics['unhealthy_as_healthy_fn']} "
            f"TP={metrics['unhealthy_correct_tp']}"
        )


        print()


        del metadata


    metrics_df = pd.DataFrame(
        metric_rows
    )


    confusion_df = pd.DataFrame(
        confusion_rows
    )


    metrics_df = (
        metrics_df
        .sort_values(
            [
                "macro_f1",
                "balanced_accuracy",
                "unhealthy_recall",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )


    metrics_df.to_csv(
        METRICS_CSV,
        index=False,
    )


    confusion_df.to_csv(
        CONFUSION_CSV,
        index=False,
    )


    print()
    print("=" * 92)
    print(
        " LIVE EVALUATION SUMMARY"
    )
    print("=" * 92)


    summary_columns = [
        "model",
        "samples",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "unhealthy_precision",
        "unhealthy_recall",
        "unhealthy_f1",
        "healthy_as_unhealthy_fp",
        "unhealthy_as_healthy_fn",
        "unhealthy_correct_tp",
    ]


    print(
        metrics_df[
            summary_columns
        ].to_string(
            index=False
        )
    )


    print()


    if (
        "Old_MobileNetV2"
        in metrics_df[
            "model"
        ].values

        and

        "New_MobileNetV2"
        in metrics_df[
            "model"
        ].values
    ):

        old_row = (
            metrics_df[
                metrics_df[
                    "model"
                ]
                == "Old_MobileNetV2"
            ]
            .iloc[0]
        )


        new_row = (
            metrics_df[
                metrics_df[
                    "model"
                ]
                == "New_MobileNetV2"
            ]
            .iloc[0]
        )


        recall_change_pp = (
            new_row[
                "unhealthy_recall"
            ]
            -
            old_row[
                "unhealthy_recall"
            ]
        ) * 100.0


        f1_change_pp = (
            new_row[
                "unhealthy_f1"
            ]
            -
            old_row[
                "unhealthy_f1"
            ]
        ) * 100.0


        print(
            "OLD -> NEW MobileNetV2 LIVE CHANGE"
        )


        print(
            f"Unhealthy recall change : "
            f"{recall_change_pp:+.2f} percentage points"
        )


        print(
            f"Unhealthy F1 change     : "
            f"{f1_change_pp:+.2f} percentage points"
        )


        print()


    print(
        "IMPORTANT:"
    )


    print(
        "This evaluation uses independent manual labels and "
        "does not modify the production database predictions."
    )


    print(
        "Samples from the same capture are not statistically "
        "independent because they share capture-level conditions. "
        "Use grouped/capture-aware statistical analysis later."
    )


    print()


    print(
        "Predictions CSV:"
    )

    print(
        PREDICTIONS_CSV
    )


    print(
        "Metrics CSV:"
    )

    print(
        METRICS_CSV
    )


    print(
        "Confusion CSV:"
    )

    print(
        CONFUSION_CSV
    )


    print("=" * 92)


if __name__ == "__main__":
    main()
