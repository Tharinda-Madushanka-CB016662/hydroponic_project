#!/bin/bash

# ============================================================
# HYDROPONIC EDGE-AI AUTOMATED PIPELINE
# ============================================================
#
# This script is executed by cron every 15 minutes.
#
# Processing order:
#   1. collect_sample.py
#      - Reads ESP32 sensor values
#      - Captures the full Raspberry Pi camera image
#      - Creates P01-P12 crops
#      - Saves the capture and plant records to SQLite
#
#   2. run_inference.py
#      - Random Forest sensor inference always runs
#      - MobileNetV2 image inference runs only when image quality
#        passes the configured time + Lux checks
#      - Complementary decision-level fusion runs when image inference is available
#      - Classical 60/40 probability fusion is retained only as a research result
#
#   3. generate_shap.py
#      - Explains the Random Forest sensor prediction
#      - One SHAP explanation is generated per capture because
#        sensor values are shared by all plant positions
#
#   4. generate_gradcam.py
#      - Explains MobileNetV2 predictions for active plants
#      - Automatically skips NO PLANT / low-light / night captures
#
# IMPORTANT:
#   Collection + inference are core stages.
#   If either fails, the pipeline stops.
#
#   SHAP and Grad-CAM are explainability stages.
#   If either fails, the core capture/inference remains valid and
#   the pipeline logs a warning rather than losing the data.
# ============================================================


# ============================================================
# SHELL SAFETY / ENVIRONMENT
# ============================================================

# Treat use of an undefined variable as an error.
# We deliberately do NOT use `set -e` because stage exit codes are
# handled explicitly below so explainability failures can be non-fatal.
set -u

# Standard PATH because cron runs with a minimal environment.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

# Files created by the pipeline remain writable by the project group.
umask 002


# ============================================================
# PROJECT PATHS
# ============================================================

PYTHON="/hydroponic_project/venv/bin/python"
SQLITE3="/usr/bin/sqlite3"

DB_FILE="/hydroponic_project/database/hydroponic.db"

COLLECT_SCRIPT="/hydroponic_project/data_collection/collect_sample.py"
INFERENCE_SCRIPT="/hydroponic_project/inference/run_inference.py"
SHAP_SCRIPT="/hydroponic_project/explainability/generate_shap.py"
GRADCAM_SCRIPT="/hydroponic_project/explainability/generate_gradcam.py"

# flock prevents two cron executions from using the camera/serial port
# at the same time if a previous run has not yet finished.
LOCK_FILE="/tmp/hydroponic_pipeline.lock"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

print_separator() {
    echo "========================================"
}

# Verify required files before touching the hardware.
require_file() {
    local file_path="$1"
    local description="$2"

    if [ ! -f "$file_path" ]; then
        print_separator
        log_message "ERROR: ${description} not found"
        echo "$file_path"
        print_separator
        exit 1
    fi
}


# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

require_file "$PYTHON" "Python virtual-environment executable"
require_file "$SQLITE3" "sqlite3 executable"
require_file "$DB_FILE" "Hydroponic SQLite database"
require_file "$COLLECT_SCRIPT" "Data collection script"
require_file "$INFERENCE_SCRIPT" "Inference script"
require_file "$SHAP_SCRIPT" "SHAP explainability script"
require_file "$GRADCAM_SCRIPT" "Grad-CAM explainability script"


# ============================================================
# PREVENT OVERLAPPING PIPELINE RUNS
# ============================================================

# Open file descriptor 9 on the lock file.
exec 9>"$LOCK_FILE"

# Obtain the lock without waiting.
# If another pipeline already holds the lock, this cron run exits.
if ! /usr/bin/flock -n 9; then
    print_separator
    log_message "Pipeline already running. This scheduled run is skipped."
    print_separator
    exit 0
fi


# ============================================================
# PIPELINE START
# ============================================================

PIPELINE_START=$(date +%s)

 echo
print_separator
log_message "HYDROPONIC EDGE-AI PIPELINE START"
print_separator


# ============================================================
# STEP 1 - DATA COLLECTION
# ============================================================
#
# collect_sample.py creates ONE new capture_id and writes:
#   capture_events  -> one capture-level row
#   plant_samples   -> P01-P12 plant rows
#   images/crops    -> image files on disk
# ============================================================

echo
print_separator
log_message "STEP 1/4 - Starting data collection"
print_separator

COLLECT_START=$(date +%s)

"$PYTHON" "$COLLECT_SCRIPT"
COLLECT_STATUS=$?

COLLECT_END=$(date +%s)
COLLECT_SECONDS=$((COLLECT_END - COLLECT_START))

if [ $COLLECT_STATUS -ne 0 ]; then
    echo
    print_separator
    log_message "ERROR: Data collection failed with exit code $COLLECT_STATUS"
    log_message "Inference and explainability will NOT run."
    print_separator
    exit $COLLECT_STATUS
fi

log_message "Data collection completed successfully in ${COLLECT_SECONDS}s"


# ============================================================
# IDENTIFY THE CAPTURE CREATED BY STEP 1
# ============================================================
#
# Passing the same explicit capture_id to every following stage makes
# the pipeline deterministic. Each stage therefore processes exactly
# the capture created by this pipeline run rather than independently
# asking for whichever row happens to be latest later.
# ============================================================

CAPTURE_ID=$(
    "$SQLITE3" -noheader "$DB_FILE" \
    "SELECT capture_id FROM capture_events ORDER BY id DESC LIMIT 1;"
)

# Remove possible carriage-return/newline whitespace.
CAPTURE_ID=$(echo "$CAPTURE_ID" | tr -d '\r\n')

if [ -z "$CAPTURE_ID" ]; then
    echo
    print_separator
    log_message "ERROR: Could not determine the newly created capture_id"
    print_separator
    exit 1
fi

log_message "Capture selected for this pipeline: $CAPTURE_ID"


# ============================================================
# STEP 2 - AI INFERENCE
# ============================================================
#
# run_inference.py currently implements:
#   - Sensor Random Forest -> always runs
#   - Image MobileNetV2    -> only if time + Lux quality checks pass
#   - Complementary decision -> combines visible-health + environmental state
#   - Classical late fusion   -> not used by final production decision
#   - Active/Inactive      -> NO PLANT positions are skipped
# ============================================================

echo
print_separator
log_message "STEP 2/4 - Starting final sensor/image/complementary-decision inference"
print_separator

INFERENCE_START=$(date +%s)

"$PYTHON" "$INFERENCE_SCRIPT" \
    --capture-id "$CAPTURE_ID"

INFERENCE_STATUS=$?

INFERENCE_END=$(date +%s)
INFERENCE_SECONDS=$((INFERENCE_END - INFERENCE_START))

if [ $INFERENCE_STATUS -ne 0 ]; then
    echo
    print_separator
    log_message "ERROR: AI inference failed with exit code $INFERENCE_STATUS"
    log_message "SHAP and Grad-CAM will NOT run for this capture."
    print_separator
    exit $INFERENCE_STATUS
fi

log_message "AI inference completed successfully in ${INFERENCE_SECONDS}s"


# ============================================================
# STEP 3 - SENSOR SHAP EXPLAINABILITY
# ============================================================
#
# SHAP explains the capture-level 5-feature Random Forest environmental prediction.
# Water level is intentionally excluded from the RF and remains a direct alert.
# It runs during both daytime and nighttime because sensor inference is
# independent of camera lighting.
#
# A SHAP failure does NOT invalidate the collected data or prediction.
# ============================================================

echo
print_separator
log_message "STEP 3/4 - Starting Random Forest SHAP explanation"
print_separator

SHAP_START=$(date +%s)

"$PYTHON" "$SHAP_SCRIPT" \
    --capture-id "$CAPTURE_ID"

SHAP_STATUS=$?

SHAP_END=$(date +%s)
SHAP_SECONDS=$((SHAP_END - SHAP_START))

if [ $SHAP_STATUS -ne 0 ]; then
    echo
    print_separator
    log_message "WARNING: SHAP generation failed with exit code $SHAP_STATUS"
    log_message "Collection and AI inference remain valid."
    print_separator
else
    log_message "SHAP explanation completed successfully in ${SHAP_SECONDS}s"
fi


# ============================================================
# STEP 4 - IMAGE GRAD-CAM EXPLAINABILITY
# ============================================================
#
# Grad-CAM is plant/image-level explainability.
# generate_gradcam.py decides which plants are eligible:
#   plant_active = 1 + real image prediction -> generate heatmap
#   NO PLANT / unknown / image skipped       -> no heatmap
#
# Therefore this stage can safely be called for every capture.
# ============================================================

echo
print_separator
log_message "STEP 4/4 - Starting MobileNetV2 Grad-CAM explanation"
print_separator

GRADCAM_START=$(date +%s)

"$PYTHON" "$GRADCAM_SCRIPT" \
    --capture-id "$CAPTURE_ID"

GRADCAM_STATUS=$?

GRADCAM_END=$(date +%s)
GRADCAM_SECONDS=$((GRADCAM_END - GRADCAM_START))

if [ $GRADCAM_STATUS -ne 0 ]; then
    echo
    print_separator
    log_message "WARNING: Grad-CAM generation failed with exit code $GRADCAM_STATUS"
    log_message "Collection, inference and any successful SHAP result remain valid."
    print_separator
else
    log_message "Grad-CAM stage completed successfully in ${GRADCAM_SECONDS}s"
fi


# ============================================================
# PIPELINE SUMMARY
# ============================================================

PIPELINE_END=$(date +%s)
PIPELINE_SECONDS=$((PIPELINE_END - PIPELINE_START))

 echo
print_separator
log_message "HYDROPONIC EDGE-AI PIPELINE COMPLETED"
echo "Capture ID       : $CAPTURE_ID"
echo "Collection       : SUCCESS (${COLLECT_SECONDS}s)"
echo "AI inference     : SUCCESS (${INFERENCE_SECONDS}s)"

if [ $SHAP_STATUS -eq 0 ]; then
    echo "Sensor SHAP      : SUCCESS (${SHAP_SECONDS}s)"
else
    echo "Sensor SHAP      : WARNING / FAILED"
fi

if [ $GRADCAM_STATUS -eq 0 ]; then
    echo "Image Grad-CAM   : SUCCESS (${GRADCAM_SECONDS}s)"
else
    echo "Image Grad-CAM   : WARNING / FAILED"
fi

echo "Total pipeline   : ${PIPELINE_SECONDS}s"
print_separator
 echo

# Core collection + inference were successful, so return success even if
# an optional explainability stage logged a warning.
exit 0
