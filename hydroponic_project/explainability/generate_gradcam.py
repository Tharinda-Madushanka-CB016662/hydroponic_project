import argparse
import sqlite3
import sys

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

DB_FILE = Path(
    "/hydroponic_project/database/hydroponic.db"
)

MODEL_CANDIDATES = [
    Path(
        "/hydroponic_project/models/image/production/"
        "mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth"
    ),
    Path(
        "/hydroponic_project/models/image/candidates/"
        "mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth"
    ),
]

GRADCAM_ROOT = Path(
    "/hydroponic_project/explainability/gradcam"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# ============================================================
# MODEL
# ============================================================

class Model2Class(nn.Module):

    def __init__(
        self,
        num_classes=2
    ):

        super().__init__()

        self.model = models.mobilenet_v2(
            weights=None
        )

        in_features = (
            self.model
            .classifier[1]
            .in_features
        )

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(
        self,
        x
    ):

        return self.model(x)


# ============================================================
# PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ]
)


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = (
            target_layer
        )

        self.activations = None

        self.gradients = None

        self.forward_hook = (
            target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

        if output.requires_grad:

            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(
        self,
        gradient
    ):

        self.gradients = gradient

    def generate(
        self,
        input_tensor,
        class_idx
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_idx
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations "
                "were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients "
                "were not captured."
            )

        weights = (
            self.gradients
            .mean(
                dim=(2, 3),
                keepdim=True
            )
        )

        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(
                input_tensor.shape[2],
                input_tensor.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        cam = (
            cam
            - cam.min()
        )

        maximum = cam.max()

        if maximum > 0:

            cam = (
                cam
                / maximum
            )

        return cam

    def remove_hooks(
        self
    ):

        self.forward_hook.remove()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model_file = None

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model_file = candidate
            break

    if model_file is None:
        checked = "\n".join(
            f"  - {candidate}"
            for candidate in MODEL_CANDIDATES
        )

        raise FileNotFoundError(
            "Final MobileNetV2 model not found. Checked:\n"
            + checked
        )

    checkpoint = torch.load(
        model_file,
        map_location=DEVICE,
        weights_only=False
    )

    num_classes = checkpoint.get(
        "num_classes",
        2
    )

    model = Model2Class(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        DEVICE
    )

    model.eval()

    class_names = checkpoint[
        "class_names"
    ]

    class_to_idx = checkpoint[
        "class_to_idx"
    ]

    print(
        f"Model file  : "
        f"{model_file}"
    )

    print(
        f"Model       : "
        f"{checkpoint.get('model_name')}"
    )

    print(
        f"Classes     : "
        f"{class_names}"
    )

    print(
        f"Class map   : "
        f"{class_to_idx}"
    )

    return (
        model,
        class_names,
        class_to_idx
    )


# ============================================================
# GET CAPTURE
# ============================================================

def get_capture(
    capture_id=None
):

    conn = sqlite3.connect(
        str(DB_FILE)
    )

    conn.row_factory = sqlite3.Row

    try:

        cursor = conn.cursor()

        if capture_id:

            cursor.execute(
                """
                SELECT *
                FROM capture_events
                WHERE capture_id = ?
                """,
                (
                    capture_id,
                )
            )

        else:

            cursor.execute(
                """
                SELECT *
                FROM capture_events
                ORDER BY id DESC
                LIMIT 1
                """
            )

        capture = cursor.fetchone()

        if capture is None:

            raise RuntimeError(
                "Capture not found."
            )

        actual_capture_id = (
            capture[
                "capture_id"
            ]
        )

        cursor.execute(
            """
            SELECT

                id,
                capture_id,
                plant_id,

                plant_active,

                crop_path,
                model_image_path,

                image_prediction,
                image_confidence,

                final_status

            FROM plant_samples

            WHERE capture_id = ?

            ORDER BY plant_id
            """,
            (
                actual_capture_id,
            )
        )

        plants = [
            dict(row)
            for row in cursor.fetchall()
        ]

        return (
            dict(capture),
            plants
        )

    finally:

        conn.close()


# ============================================================
# CHECK WHETHER GRAD-CAM IS ALLOWED
# ============================================================

def should_generate_gradcam(
    plant
):

    # --------------------------------------------------------
    # Must contain a plant
    # --------------------------------------------------------

    if plant[
        "plant_active"
    ] != 1:

        return (
            False,
            "NO ACTIVE PLANT"
        )

    prediction = str(
        plant.get(
            "image_prediction"
        )
        or ""
    ).strip().lower()

    # --------------------------------------------------------
    # Image inference must actually have run
    # --------------------------------------------------------

    if not prediction:

        return (
            False,
            "NO IMAGE PREDICTION"
        )

    if prediction.startswith(
        "skipped"
    ):

        return (
            False,
            "IMAGE INFERENCE SKIPPED"
        )

    if prediction in {
        "no plant",
        "occupancy unknown"
    }:

        return (
            False,
            prediction.upper()
        )

    # --------------------------------------------------------
    # Valid MobileNet classes
    # --------------------------------------------------------

    if prediction not in {
        "healthy",
        "unhealthy"
    }:

        return (
            False,
            f"UNSUPPORTED PREDICTION: "
            f"{prediction}"
        )

    return (
        True,
        "OK"
    )


# ============================================================
# GENERATE OVERLAY
# ============================================================

def create_overlay(
    image_path,
    cam,
    output_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Unable to read image: "
            f"{image_path}"
        )

    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.55,
        heatmap,
        0.45,
        0
    )

    cv2.imwrite(
        str(output_path),
        overlay,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


# ============================================================
# UPDATE DATABASE
# ============================================================

def update_gradcam_path(
    row_id,
    gradcam_path
):

    conn = sqlite3.connect(
        str(DB_FILE),
        timeout=30
    )

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE plant_samples

            SET gradcam_path = ?

            WHERE id = ?
            """,
            (
                str(
                    gradcam_path
                ),
                row_id
            )
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
        "--capture-id",
        default=None,
        help=(
            "Capture ID to explain. "
            "Default: latest capture."
        )
    )

    args = parser.parse_args()

    print()
    print(
        "========================================"
    )
    print(
        " HYDROPONIC GRAD-CAM EXPLAINABILITY"
    )
    print(
        "========================================"
    )

    (
        capture,
        plants
    ) = get_capture(
        args.capture_id
    )

    capture_id = capture[
        "capture_id"
    ]

    print(
        f"Capture ID : "
        f"{capture_id}"
    )

    print(
        f"Timestamp  : "
        f"{capture['timestamp']}"
    )

    output_dir = (
        GRADCAM_ROOT
        / capture_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Determine whether at least one plant requires Grad-CAM
    # --------------------------------------------------------

    eligible_plants = []

    for plant in plants:

        allowed, reason = (
            should_generate_gradcam(
                plant
            )
        )

        if allowed:

            eligible_plants.append(
                plant
            )

    if not eligible_plants:

        print()
        print(
            "No plants are eligible "
            "for Grad-CAM."
        )

        print(
            "Image inference may have "
            "been skipped for this capture."
        )

        return

    # --------------------------------------------------------
    # Load MobileNetV2 only if needed
    # --------------------------------------------------------

    (
        model,
        class_names,
        class_to_idx
    ) = load_model()

    # Last MobileNetV2 convolutional feature block
    target_layer = (
        model
        .model
        .features[-1]
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    generated = 0

    skipped = 0

    try:

        for plant in plants:

            plant_id = (
                plant[
                    "plant_id"
                ]
            )

            allowed, reason = (
                should_generate_gradcam(
                    plant
                )
            )

            print()
            print(
                f"{plant_id}"
            )

            if not allowed:

                print(
                    f"  Grad-CAM : SKIPPED"
                )

                print(
                    f"  Reason   : {reason}"
                )

                skipped += 1

                continue

            image_path = Path(
                plant[
                    "model_image_path"
                ]
            )

            if not image_path.exists():

                print(
                    "  Grad-CAM : FAILED"
                )

                print(
                    "  Reason   : "
                    "Image file missing"
                )

                skipped += 1

                continue

            prediction = str(
                plant[
                    "image_prediction"
                ]
            ).lower()

            # The stored image_prediction comes from the deployed
            # MobileNetV2 threshold decision (0.42), so Grad-CAM explains
            # that deployed class even when it differs from simple argmax.
            class_idx = (
                class_to_idx[
                    prediction
                ]
            )

            image = (
                Image.open(
                    image_path
                )
                .convert(
                    "RGB"
                )
            )

            tensor = IMAGE_TRANSFORM(
                image
            )

            tensor = (
                tensor
                .unsqueeze(
                    0
                )
                .to(
                    DEVICE
                )
            )

            cam = gradcam.generate(
                tensor,
                class_idx
            )

            output_path = (
                output_dir
                /
                f"{plant_id}_gradcam.jpg"
            )

            create_overlay(
                image_path,
                cam,
                output_path
            )

            update_gradcam_path(
                plant[
                    "id"
                ],
                output_path
            )

            print(
                f"  Prediction: "
                f"{prediction}"
            )

            print(
                f"  Grad-CAM  : "
                f"{output_path}"
            )

            generated += 1

    finally:

        gradcam.remove_hooks()

    print()
    print(
        "========================================"
    )
    print(
        " GRAD-CAM COMPLETED"
    )
    print(
        "========================================"
    )

    print(
        f"Generated : {generated}"
    )

    print(
        f"Skipped   : {skipped}"
    )

    print(
        f"Output    : {output_dir}"
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
        print(
            "========================================"
        )
        print(
            " GRAD-CAM FAILED"
        )
        print(
            "========================================"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        sys.exit(1)
