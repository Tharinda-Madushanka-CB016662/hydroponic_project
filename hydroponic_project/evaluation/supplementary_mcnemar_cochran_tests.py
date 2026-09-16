#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

FINAL_COMPARISON_CSV = Path(
    "/hydroponic_project/evaluation/final_image_sensor_fusion_comparison/"
    "final_observation_predictions.csv"
)

IMAGE_MODEL_CSV = Path(
    "/hydroponic_project/evaluation/live_model_evaluation/"
    "live_image_model_predictions.csv"
)

OUTPUT_DIR = Path(
    "/hydroponic_project/evaluation/supplementary_paired_tests"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH_COLUMN = "ground_truth"
DEFAULT_CNN_THRESHOLD = 0.50


def normalise_label_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def exact_mcnemar(
    y_true,
    prediction_a,
    prediction_b,
    name_a,
    name_b,
):
    y_true = np.asarray(y_true)
    prediction_a = np.asarray(prediction_a)
    prediction_b = np.asarray(prediction_b)

    correct_a = prediction_a == y_true
    correct_b = prediction_b == y_true

    both_correct = int(np.sum(correct_a & correct_b))
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                k=min(a_correct_b_wrong, a_wrong_b_correct),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "model_a": name_a,
        "model_b": name_b,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "exact_mcnemar_p_value": p_value,
    }


def cochran_q_test(correctness_matrix):
    matrix = np.asarray(correctness_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Correctness matrix must be two-dimensional.")

    n, k = matrix.shape

    if k < 2:
        raise ValueError("Cochran's Q requires at least two classifiers.")

    column_totals = matrix.sum(axis=0)
    row_totals = matrix.sum(axis=1)
    total = float(column_totals.sum())

    numerator = (
        (k - 1)
        * (
            k * np.sum(column_totals ** 2)
            - total ** 2
        )
    )

    denominator = (
        k * total
        - np.sum(row_totals ** 2)
    )

    if denominator <= 0:
        q_statistic = np.nan
        p_value = np.nan
    else:
        q_statistic = float(numerator / denominator)
        p_value = float(
            chi2.sf(
                q_statistic,
                df=k - 1,
            )
        )

    return {
        "n_observations": int(n),
        "n_classifiers": int(k),
        "q_statistic": q_statistic,
        "degrees_of_freedom": int(k - 1),
        "p_value": p_value,
    }


def holm_adjust(dataframe, p_column):
    result = dataframe.copy()

    p_values = result[p_column].astype(float).to_numpy()
    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, p_value in enumerate(sorted_p):
        multiplier = m - rank
        adjusted = min(1.0, multiplier * p_value)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted

    result["holm_adjusted_p_value"] = adjusted
    result["significant_after_holm_0_05"] = (
        result["holm_adjusted_p_value"] < 0.05
    )

    return result


# -------------------------------------------------------------------------
# PART A - IMAGE ONLY vs FUSION
# -------------------------------------------------------------------------

if not FINAL_COMPARISON_CSV.exists():
    raise FileNotFoundError(
        f"Missing final comparison CSV:\n{FINAL_COMPARISON_CSV}"
    )

fusion_df = pd.read_csv(FINAL_COMPARISON_CSV)

required_fusion_columns = {
    "ground_truth",
    "E1_image_prediction",
    "E3_fusion_60_40_prediction",
    "E3_best_exploratory_prediction",
    "capture_id",
}

missing = required_fusion_columns - set(fusion_df.columns)

if missing:
    raise RuntimeError(
        "Missing columns in final comparison CSV: "
        f"{sorted(missing)}"
    )

for column in [
    "ground_truth",
    "E1_image_prediction",
    "E3_fusion_60_40_prediction",
    "E3_best_exploratory_prediction",
]:
    fusion_df[column] = normalise_label_series(fusion_df[column])

fusion_df = (
    fusion_df[
        fusion_df["ground_truth"].isin(["healthy", "unhealthy"])
    ]
    .copy()
    .reset_index(drop=True)
)

image_vs_fusion_60 = exact_mcnemar(
    y_true=fusion_df["ground_truth"],
    prediction_a=fusion_df["E1_image_prediction"],
    prediction_b=fusion_df["E3_fusion_60_40_prediction"],
    name_a="New MobileNetV2 Image-Only",
    name_b="Classical Fusion 60/40",
)

image_vs_best_fusion = exact_mcnemar(
    y_true=fusion_df["ground_truth"],
    prediction_a=fusion_df["E1_image_prediction"],
    prediction_b=fusion_df["E3_best_exploratory_prediction"],
    name_a="New MobileNetV2 Image-Only",
    name_b="Best Exploratory Fusion",
)

fusion_mcnemar_df = pd.DataFrame(
    [image_vs_fusion_60, image_vs_best_fusion]
)

fusion_mcnemar_df.to_csv(
    OUTPUT_DIR / "image_vs_fusion_exact_mcnemar.csv",
    index=False,
)


# -------------------------------------------------------------------------
# PART B - CNN ARCHITECTURE COMPARISON
# -------------------------------------------------------------------------

if not IMAGE_MODEL_CSV.exists():
    raise FileNotFoundError(
        f"Missing image-model CSV:\n{IMAGE_MODEL_CSV}"
    )

cnn_df = pd.read_csv(IMAGE_MODEL_CSV)

required_cnn_columns = {
    "ground_truth",
    "capture_id",
    "plant_id",
    "New_MobileNetV2_unhealthy_probability",
    "New_ResNet18_unhealthy_probability",
    "New_EfficientNet_B0_unhealthy_probability",
}

missing = required_cnn_columns - set(cnn_df.columns)

if missing:
    raise RuntimeError(
        "Missing columns in image-model CSV: "
        f"{sorted(missing)}"
    )

cnn_df["ground_truth"] = normalise_label_series(
    cnn_df["ground_truth"]
)

cnn_df = (
    cnn_df[
        cnn_df["ground_truth"].isin(["healthy", "unhealthy"])
    ]
    .copy()
    .reset_index(drop=True)
)

cnn_probability_columns = {
    "MobileNetV2":
        "New_MobileNetV2_unhealthy_probability",
    "ResNet18":
        "New_ResNet18_unhealthy_probability",
    "EfficientNet-B0":
        "New_EfficientNet_B0_unhealthy_probability",
}

for probability_column in cnn_probability_columns.values():
    cnn_df[probability_column] = pd.to_numeric(
        cnn_df[probability_column],
        errors="coerce",
    )

cnn_df = cnn_df.dropna(
    subset=list(cnn_probability_columns.values())
).copy()

for model_name, probability_column in cnn_probability_columns.items():
    cnn_df[f"{model_name}_prediction_050"] = np.where(
        cnn_df[probability_column] >= DEFAULT_CNN_THRESHOLD,
        "unhealthy",
        "healthy",
    )

    cnn_df[f"{model_name}_correct"] = (
        cnn_df[f"{model_name}_prediction_050"]
        == cnn_df["ground_truth"]
    ).astype(int)

model_names = [
    "MobileNetV2",
    "ResNet18",
    "EfficientNet-B0",
]

correctness_matrix = np.column_stack(
    [
        cnn_df[f"{model_name}_correct"].to_numpy()
        for model_name in model_names
    ]
)

cochran_result = cochran_q_test(correctness_matrix)
cochran_result["models"] = " | ".join(model_names)
cochran_result["threshold_for_all_models"] = DEFAULT_CNN_THRESHOLD

cochran_df = pd.DataFrame([cochran_result])

cochran_df.to_csv(
    OUTPUT_DIR / "cnn_cochran_q.csv",
    index=False,
)

pairwise_pairs = [
    ("MobileNetV2", "ResNet18"),
    ("MobileNetV2", "EfficientNet-B0"),
    ("ResNet18", "EfficientNet-B0"),
]

pairwise_rows = []

for model_a, model_b in pairwise_pairs:
    result = exact_mcnemar(
        y_true=cnn_df["ground_truth"],
        prediction_a=cnn_df[f"{model_a}_prediction_050"],
        prediction_b=cnn_df[f"{model_b}_prediction_050"],
        name_a=model_a,
        name_b=model_b,
    )
    pairwise_rows.append(result)

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df = holm_adjust(
    pairwise_df,
    p_column="exact_mcnemar_p_value",
)

pairwise_df.to_csv(
    OUTPUT_DIR / "cnn_pairwise_exact_mcnemar_holm.csv",
    index=False,
)

cnn_summary_rows = []

for model_name in model_names:
    correct = int(
        cnn_df[f"{model_name}_correct"].sum()
    )
    total = int(len(cnn_df))

    cnn_summary_rows.append(
        {
            "model": model_name,
            "threshold": DEFAULT_CNN_THRESHOLD,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": correct / total,
        }
    )

cnn_summary_df = pd.DataFrame(cnn_summary_rows)

cnn_summary_df.to_csv(
    OUTPUT_DIR / "cnn_default_050_correctness_summary.csv",
    index=False,
)


# -------------------------------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------------------------------

print()
print("=" * 100)
print(" SUPPLEMENTARY PAIRED STATISTICAL TESTS")
print("=" * 100)

print("Visible-labelled observations :", len(cnn_df))
print(
    "Capture clusters              :",
    cnn_df["capture_id"].nunique(),
)

print()
print("=" * 100)
print(" A. EXACT McNEMAR: IMAGE-ONLY vs CLASSICAL FUSION")
print("=" * 100)
print(fusion_mcnemar_df.to_string(index=False))

print()
print(
    "a_correct_b_wrong = Image correct, Fusion wrong."
)
print(
    "a_wrong_b_correct = Image wrong, Fusion correct."
)

print()
print("=" * 100)
print(" B. CNN ARCHITECTURE ACCURACY AT COMMON THRESHOLD 0.50")
print("=" * 100)
print(cnn_summary_df.to_string(index=False))

print()
print("=" * 100)
print(" C. COCHRAN'S Q OMNIBUS TEST")
print("=" * 100)
print(cochran_df.to_string(index=False))

print()

if (
    not pd.isna(cochran_result["p_value"])
    and cochran_result["p_value"] < 0.05
):
    print(
        "Cochran's Q is significant at alpha = 0.05."
    )
    print(
        "At least one CNN has a different probability of correct classification."
    )
else:
    print(
        "Cochran's Q is NOT significant at alpha = 0.05."
    )
    print(
        "There is insufficient evidence of a difference among the three CNNs."
    )

print()
print("=" * 100)
print(" D. PAIRWISE EXACT McNEMAR + HOLM CORRECTION")
print("=" * 100)
print(pairwise_df.to_string(index=False))

print()
print("IMPORTANT:")
print(
    "McNemar and Cochran's Q are supplementary because observations "
    "are clustered within capture events."
)
print(
    "The capture-cluster bootstrap and paired capture-level permutation "
    "analysis remain the primary inferential evidence for Image vs Fusion."
)
print(
    "All three CNNs use threshold 0.50 here so the architecture comparison "
    "is fair and is not biased by MobileNetV2's later tuned threshold 0.42."
)

print()
print("Output directory:")
print(OUTPUT_DIR)
print("=" * 100)
