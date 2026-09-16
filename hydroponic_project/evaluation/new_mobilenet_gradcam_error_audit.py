#!/usr/bin/env python3

# ============================================================
# NEW MOBILENETV2 LIVE ERROR + GRAD-CAM AUDIT
# ============================================================
#
# Purpose
# -------
# Audit the NEW MobileNetV2 mistakes on manually labelled
# Raspberry Pi observations using the exploratory threshold
# selected from the sensitivity analysis.
#
# Default exploratory threshold:
#     0.42
#
# This script:
#   1. Reads the existing live prediction CSV.
#   2. Recomputes Healthy/Unhealthy decisions at threshold 0.42.
#   3. Identifies false positives and false negatives.
#   4. Summarises errors by plant position.
#   5. Generates Grad-CAM overlays ONLY for error observations.
#   6. Saves CSV evidence for dissertation error analysis.
#
# IMPORTANT:
#   - READ-ONLY with respect to SQLite.
#   - Does NOT replace the production model.
#   - Does NOT alter stored predictions/fusion results.
#   - 0.42 is exploratory, not yet independently validated.
# ============================================================

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PREDICTIONS_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)

MODEL_PATH = Path(
    "/hydroponic_project/models/image/candidates/"
    "mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth"
)

OUTPUT_ROOT = Path(
    "/hydroponic_project/evaluation/"
    "new_mobilenet_gradcam_error_audit"
)

ALL_AUDIT_CSV = (
    OUTPUT_ROOT
    / "new_mobilenet_threshold_042_all_observations.csv"
)

ERRORS_CSV = (
    OUTPUT_ROOT
    / "new_mobilenet_threshold_042_errors.csv"
)

PLANT_SUMMARY_CSV = (
    OUTPUT_ROOT
    / "new_mobilenet_threshold_042_error_by_plant.csv"
)

GRADCAM_DIR = (
    OUTPUT_ROOT
    / "gradcam_errors"
)


# ============================================================
# MODEL / THRESHOLD SETTINGS
# ============================================================

DEFAULT_THRESHOLD = 0.42

CLASS_NAMES = [
    "healthy",
    "unhealthy",
]

EXPECTED_MAPPING = {
    "healthy": 0,
    "unhealthy": 1,
}

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

PYTORCH_THREADS = 4


# ============================================================
# PREPROCESSING
# ============================================================

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
        num_classes=2,
    ):

        super().__init__()

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


    def forward(
        self,
        x,
    ):

        return self.model(
            x
        )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            self.target_layer
            .register_forward_hook(
                self._save_activation
            )
        )


    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = (
            output.detach()
        )

        output.register_hook(
            self._save_gradient
        )


    def _save_gradient(
        self,
        gradient,
    ):

        self.gradients = (
            gradient.detach()
        )


    def generate(
        self,
        input_tensor,
        class_idx,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        score = output[
            :,
            class_idx
        ].sum()

        score.backward()


        if (
            self.activations is None
            or self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "activations/gradients."
            )


        weights = (
            self.gradients
            .mean(
                dim=(
                    2,
                    3,
                ),
                keepdim=True,
            )
        )


        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )


        cam = F.relu(
            cam
        )


        cam = F.interpolate(
            cam,
            size=input_tensor.shape[
                2:
            ],
            mode="bilinear",
            align_corners=False,
        )


        cam = (
            cam[
                0,
                0
            ]
            .cpu()
            .numpy()
        )


        cam = (
            cam
            - cam.min()
        )


        max_value = (
            cam.max()
        )


        if max_value > 0:

            cam = (
                cam
                / max_value
            )


        return cam


    def remove_hooks(
        self,
    ):

        self.forward_hook.remove()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )


    mapping = checkpoint.get(
        "class_to_idx",
        EXPECTED_MAPPING,
    )


    if mapping != EXPECTED_MAPPING:

        raise RuntimeError(
            f"Unexpected class mapping: "
            f"{mapping}"
        )


    model = Model2Class(
        num_classes=2
    ).to(
        DEVICE
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    return model


# ============================================================
# LOAD / CLASSIFY AUDIT OBSERVATIONS
# ============================================================

def load_audit_dataframe(
    threshold,
):

    if not PREDICTIONS_CSV.exists():

        raise FileNotFoundError(
            f"Predictions CSV not found: "
            f"{PREDICTIONS_CSV}"
        )


    df = pd.read_csv(
        PREDICTIONS_CSV
    )


    required = {
        "capture_id",
        "plant_id",
        "model_image_path",
        "ground_truth",
        "New_MobileNetV2_unhealthy_probability",
    }


    missing = (
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )


    df[
        "ground_truth"
    ] = (
        df[
            "ground_truth"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )


    df = (
        df[
            df[
                "ground_truth"
            ].isin(
                [
                    "healthy",
                    "unhealthy",
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    probability_column = (
        "New_MobileNetV2_unhealthy_probability"
    )


    df[
        probability_column
    ] = pd.to_numeric(
        df[
            probability_column
        ],
        errors="coerce",
    )


    if df[
        probability_column
    ].isna().any():

        raise RuntimeError(
            "Invalid New MobileNetV2 probability "
            "values were found."
        )


    df[
        "threshold_used"
    ] = threshold


    df[
        "threshold_prediction"
    ] = np.where(
        df[
            probability_column
        ]
        >= threshold,
        "unhealthy",
        "healthy",
    )


    def determine_error_type(
        row,
    ):

        truth = row[
            "ground_truth"
        ]

        prediction = row[
            "threshold_prediction"
        ]


        if (
            truth == "healthy"
            and prediction == "unhealthy"
        ):

            return "FALSE_POSITIVE"


        if (
            truth == "unhealthy"
            and prediction == "healthy"
        ):

            return "FALSE_NEGATIVE"


        return "CORRECT"


    df[
        "error_type"
    ] = df.apply(
        determine_error_type,
        axis=1,
    )


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


    return df


# ============================================================
# IMAGE + OVERLAY HELPERS
# ============================================================

def prepare_image(
    image_path,
):

    pil_image = (
        Image.open(
            image_path
        )
        .convert(
            "RGB"
        )
    )


    tensor = (
        IMAGE_TRANSFORM(
            pil_image
        )
        .unsqueeze(
            0
        )
        .to(
            DEVICE
        )
    )


    rgb_array = np.array(
        pil_image
    )


    return (
        tensor,
        rgb_array,
    )


def create_overlay(
    rgb_image,
    cam,
):

    height, width = (
        rgb_image.shape[
            :2
        ]
    )


    cam_resized = cv2.resize(
        cam,
        (
            width,
            height,
        ),
        interpolation=
            cv2.INTER_LINEAR,
    )


    heatmap = (
        np.uint8(
            255
            * cam_resized
        )
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )


    original_bgr = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )


    overlay = cv2.addWeighted(
        original_bgr,
        0.55,
        heatmap,
        0.45,
        0,
    )


    return overlay


# ============================================================
# GENERATE GRAD-CAM FOR ERRORS
# ============================================================

def generate_error_gradcams(
    model,
    error_df,
):

    GRADCAM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    gradcam = GradCAM(
        model=model,
        target_layer=
            model.model.features[-1],
    )


    saved_paths = {}


    try:

        for index, row in (
            error_df.iterrows()
        ):

            image_path = Path(
                str(
                    row[
                        "model_image_path"
                    ]
                )
            )


            if not image_path.exists():

                saved_paths[
                    index
                ] = None

                continue


            (
                tensor,
                rgb_image,
            ) = prepare_image(
                image_path
            )


            prediction = (
                row[
                    "threshold_prediction"
                ]
            )


            class_idx = (
                EXPECTED_MAPPING[
                    prediction
                ]
            )


            cam = gradcam.generate(
                input_tensor=
                    tensor,

                class_idx=
                    class_idx,
            )


            overlay = create_overlay(
                rgb_image=
                    rgb_image,

                cam=
                    cam,
            )


            error_type = (
                row[
                    "error_type"
                ]
            )


            error_dir = (
                GRADCAM_DIR
                / error_type
            )


            error_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            filename = (
                f"{row['capture_id']}_"
                f"{row['plant_id']}_"
                f"{error_type}.jpg"
            )


            output_path = (
                error_dir
                / filename
            )


            ok = cv2.imwrite(
                str(
                    output_path
                ),
                overlay,
            )


            if not ok:

                raise RuntimeError(
                    f"Failed to write: "
                    f"{output_path}"
                )


            saved_paths[
                index
            ] = str(
                output_path
            )


    finally:

        gradcam.remove_hooks()


    return saved_paths


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate New MobileNetV2 live error "
            "analysis and Grad-CAM audit."
        )
    )


    parser.add_argument(
        "--threshold",
        type=float,
        default=
            DEFAULT_THRESHOLD,
        help=(
            "Exploratory unhealthy probability "
            "threshold. Default: 0.42"
        ),
    )


    args = parser.parse_args()


    threshold = float(
        args.threshold
    )


    if not (
        0.0
        < threshold
        < 1.0
    ):

        raise ValueError(
            "--threshold must be between 0 and 1."
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


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = load_audit_dataframe(
        threshold
    )


    errors_df = (
        df[
            df[
                "error_type"
            ]
            != "CORRECT"
        ]
        .copy()
    )


    false_positive_count = int(
        (
            errors_df[
                "error_type"
            ]
            == "FALSE_POSITIVE"
        ).sum()
    )


    false_negative_count = int(
        (
            errors_df[
                "error_type"
            ]
            == "FALSE_NEGATIVE"
        ).sum()
    )


    print()
    print("=" * 92)
    print(
        " NEW MOBILENETV2 LIVE ERROR + GRAD-CAM AUDIT"
    )
    print("=" * 92)


    print(
        f"Threshold              : "
        f"{threshold:.2f}"
    )


    print(
        f"Binary observations    : "
        f"{len(df)}"
    )


    print(
        f"Correct                : "
        f"{int((df['error_type'] == 'CORRECT').sum())}"
    )


    print(
        f"False positives        : "
        f"{false_positive_count}"
    )


    print(
        f"False negatives        : "
        f"{false_negative_count}"
    )


    print(
        f"Total errors           : "
        f"{len(errors_df)}"
    )


    print()


    # --------------------------------------------------------
    # ERROR SUMMARY BY PLANT
    # --------------------------------------------------------

    plant_summary = (
        df
        .groupby(
            "plant_id"
        )
        .agg(
            observations=(
                "plant_id",
                "size",
            ),

            healthy_ground_truth=(
                "ground_truth",
                lambda values:
                    int(
                        (
                            values
                            == "healthy"
                        ).sum()
                    ),
            ),

            unhealthy_ground_truth=(
                "ground_truth",
                lambda values:
                    int(
                        (
                            values
                            == "unhealthy"
                        ).sum()
                    ),
            ),

            false_positives=(
                "error_type",
                lambda values:
                    int(
                        (
                            values
                            == "FALSE_POSITIVE"
                        ).sum()
                    ),
            ),

            false_negatives=(
                "error_type",
                lambda values:
                    int(
                        (
                            values
                            == "FALSE_NEGATIVE"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )


    plant_summary[
        "total_errors"
    ] = (
        plant_summary[
            "false_positives"
        ]
        + plant_summary[
            "false_negatives"
        ]
    )


    plant_summary = (
        plant_summary
        .sort_values(
            [
                "total_errors",
                "false_negatives",
                "false_positives",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    print("=" * 92)
    print(
        " ERROR SUMMARY BY PLANT"
    )
    print("=" * 92)


    print(
        plant_summary.to_string(
            index=False
        )
    )


    print()
    print(
        "Loading New MobileNetV2 and "
        "generating Grad-CAM for errors..."
    )


    model = load_model()


    saved_gradcams = (
        generate_error_gradcams(
            model=
                model,

            error_df=
                errors_df,
        )
    )


    errors_df[
        "gradcam_path"
    ] = errors_df.index.map(
        saved_gradcams
    )


    # --------------------------------------------------------
    # SAVE EVIDENCE
    # --------------------------------------------------------

    df.to_csv(
        ALL_AUDIT_CSV,
        index=False,
    )


    errors_df.to_csv(
        ERRORS_CSV,
        index=False,
    )


    plant_summary.to_csv(
        PLANT_SUMMARY_CSV,
        index=False,
    )


    generated_count = int(
        errors_df[
            "gradcam_path"
        ]
        .notna()
        .sum()
    )


    print()
    print("=" * 92)
    print(
        " AUDIT COMPLETED"
    )
    print("=" * 92)


    print(
        f"Grad-CAM overlays generated : "
        f"{generated_count}"
    )


    print()


    print(
        "All observations CSV:"
    )

    print(
        ALL_AUDIT_CSV
    )


    print(
        "Errors CSV:"
    )

    print(
        ERRORS_CSV
    )


    print(
        "Plant summary CSV:"
    )

    print(
        PLANT_SUMMARY_CSV
    )


    print(
        "Grad-CAM directory:"
    )

    print(
        GRADCAM_DIR
    )


    print()
    print(
        "INTERPRETATION RULE:"
    )


    print(
        "A Grad-CAM hotspot indicates where the CNN "
        "focused for its prediction. It does NOT prove "
        "that the highlighted region is diseased or stressed."
    )


    print(
        "Look for whether attention falls mainly on lettuce "
        "leaf tissue or on background/pipe/pot/shadow regions."
    )


    print("=" * 92)


if __name__ == "__main__":
    main()
