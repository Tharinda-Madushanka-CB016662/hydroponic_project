HYDROPONIC LETTUCE EDGE-AI MONITORING
=====================================

An explainable multimodal Edge-AI system for hydroponic lettuce monitoring using RGB images, environmental sensor data, computer vision, machine learning, and low-cost edge hardware.

Author: Gabadage Don Tharinda Madushanka
Student ID: CB016662
University: University of Staffordshire


1. PROJECT OVERVIEW
===================

The system combines two complementary AI branches:

- Image AI:
  MobileNetV2 estimates visible lettuce condition.

- Sensor AI:
  Random Forest estimates environmental stress risk.

The final operational decision uses four complementary states:

1. Healthy / Stable
2. Early Environmental Warning
3. Visible Plant Stress - Environment Currently Normal
4. High Risk - Visible + Environmental Stress

The system also includes:

- Raspberry Pi edge deployment
- ESP32 sensor acquisition
- P01-P12 plant monitoring
- Grad-CAM image explanations
- SHAP sensor explanations
- SQLite local storage
- Streamlit dashboard
- Historical capture browser
- Plant history
- Model benchmarking
- Manual/expert evaluation labelling


2. REPOSITORY STRUCTURE
=======================

hydroponic_project/
|
|-- hydroponic_project/
|   |-- dashboard/
|   |-- data_collection/
|   |-- database/
|   |-- dissertation_final_evidence/
|   |-- evaluation/
|   |-- explainability/
|   |-- image_data/
|   |   `-- config/
|   |-- image_rois/
|   |-- inference/
|   |-- models/
|   `-- run_pipeline.sh
|
|-- .gitignore
`-- README.md


3. CURRENT PRODUCTION MODELS
============================

3.1 Image Model
---------------

Model:
MobileNetV2

Unhealthy probability threshold:
0.42

Image inference operating time:
06:00-16:59

Minimum camera light:
10 Lux

Production model path:

/hydroponic_project/models/image/production/
mobilenet_v2_best_healthy_unhealthy_combined_stress_clean.pth


3.2 Sensor Model
----------------

Model:
Random Forest

Environmental stress threshold:
0.48

Production model path:

/hydroponic_project/models/sensor/production/
sensor_model_package_expert_5features.joblib


4. HOW TO ACCESS THE PROJECT FROM ANOTHER MACHINE
=================================================

4.1 Clone the GitHub Repository
-------------------------------

Install Git and clone the repository:

git clone https://github.com/Tharinda-Madushanka-CB016662/hydroponic_project.git

Then enter the repository:

cd hydroponic_project

The repository contains another project folder also named:

hydroponic_project/

Enter it:

cd hydroponic_project

You should then see folders such as:

dashboard/
data_collection/
database/
evaluation/
explainability/
inference/
models/


5. IMPORTANT PATH CONFIGURATION
===============================

The deployed Raspberry Pi project uses absolute paths beginning with:

/hydroponic_project/

For example:

DB_FILE = Path("/hydroponic_project/database/hydroponic.db")

Therefore, on another Linux or Raspberry Pi machine, the easiest method is to create a symbolic link.

From the OUTER cloned repository directory, run:

sudo ln -s "$(pwd)/hydroponic_project" /hydroponic_project

Check the link:

ls -l /hydroponic_project

This allows the existing Python scripts to continue using:

/hydroponic_project/...

without changing every source file.

IMPORTANT:
If /hydroponic_project already exists, inspect it before creating the symbolic link.

Alternative:
Update the absolute project paths in the Python scripts to match the new machine.


6. PYTHON VIRTUAL ENVIRONMENT
=============================

Create a Python virtual environment:

python3 -m venv venv

Activate it:

source venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip

Install the main software dependencies:

pip install streamlit pandas numpy pillow torch torchvision scikit-learn joblib shap matplotlib

Additional Raspberry Pi, GPIO, I2C, ADC and camera-specific libraries may be required for live hardware operation.


7. REQUIREMENTS.TXT
===================

For reproducibility, it is strongly recommended to add a requirements.txt file to the repository.

On the Raspberry Pi where the project is already working, activate the working virtual environment:

source /hydroponic_project/venv/bin/activate

Then create the requirements file:

pip freeze > requirements.txt

Copy or move requirements.txt into the Git repository root and commit it.

Another user can then install dependencies with:

pip install -r requirements.txt


8. RUNNING THE STREAMLIT DASHBOARD
==================================

Activate the virtual environment:

source venv/bin/activate

Start the dashboard:

streamlit run /hydroponic_project/dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501

On the Raspberry Pi itself, open:

http://localhost:8501

From another computer on the same network, open:

http://<RASPBERRY_PI_IP>:8501

Example:

http://192.168.1.136:8501

To find the Raspberry Pi IP address:

hostname -I


9. RUNNING THE MAIN PIPELINE
============================

Activate the virtual environment:

source venv/bin/activate

Make the launcher executable if required:

chmod +x /hydroponic_project/run_pipeline.sh

Run the main pipeline:

/hydroponic_project/run_pipeline.sh

The pipeline performs:

Sensor collection
        |
        v
Camera capture
        |
        v
P01-P12 ROI processing
        |
        v
Sensor AI
        |
        v
MobileNetV2 Image AI
        |
        v
Complementary Decision
        |
        v
Grad-CAM / SHAP
        |
        v
SQLite storage
        |
        v
Streamlit dashboard


10. DATABASE
============

The project uses a local SQLite database:

/hydroponic_project/database/hydroponic.db

The database contains information such as:

- capture events
- sensor measurements
- plant observations
- active/inactive plant positions
- image predictions
- sensor predictions
- complementary decision states
- historical classical-fusion results
- Grad-CAM paths
- SHAP paths
- system settings
- manual evaluation labels


11. MODEL DEPLOYMENT STRUCTURE
==============================

models/
|
|-- image/
|   |-- candidates/
|   `-- production/
|
`-- sensor/
    |-- candidates/
    `-- production/

The intended model lifecycle is:

Training
   |
   v
Candidate Model
   |
   v
Evaluation / Validation
   |
   v
Production Model

The inference pipeline prioritises validated production models.

Candidate models are retained for evaluation, testing, or fallback purposes.


12. RUNNING WITHOUT RASPBERRY PI HARDWARE
=========================================

The complete live pipeline requires:

- Raspberry Pi
- ESP32
- RGB camera
- pH sensor
- TDS/EC sensor
- temperature sensors
- humidity sensor
- water-level sensor

However, the project can still be used on a normal computer to:

- inspect the source code
- inspect the SQLite database
- review evaluation results
- inspect trained models
- run supported offline analysis
- view historical dashboard data
- inspect Grad-CAM outputs
- inspect SHAP outputs

For dashboard-only use, make sure that:

1. the SQLite database is available
2. the image files referenced by the database are available
3. the Grad-CAM and SHAP files referenced by the database are available
4. the project paths are configured correctly

Live sensor and camera collection will not work unless the required hardware is connected.


13. DASHBOARD PAGES
===================

The Streamlit dashboard includes:

Dashboard
---------
Displays the latest capture, environmental conditions, AI availability, sensor status, SHAP explanation and P01-P12 results.

Plant Health
------------
Displays the latest plant-level results and Grad-CAM evidence.

Plant History
-------------
Displays historical AI outputs for an individual plant position.

Capture Browser
---------------
Allows inspection of previous capture events.

History
-------
Displays sensor time-series monitoring.

Model Comparison
----------------
Displays Raspberry Pi image-model benchmark results.

Evaluation Labelling
--------------------
Supports blinded manual/expert labelling of historical plant images before revealing AI predictions.

Settings
--------
Allows management of:

- active/inactive plant positions
- image AI operating time
- minimum camera Lux
- image classification threshold
- sensor stress threshold
- environmental reference ranges

Settings apply to future inference only.
Historical results are preserved unless they are explicitly reprocessed.


14. EXPLAINABLE AI
==================

14.1 Grad-CAM
-------------

Grad-CAM visualises image regions that contributed to MobileNetV2 visible-health predictions.

Grad-CAM provides model-attention evidence and should not be interpreted as proof of biological causation.


14.2 SHAP
---------

SHAP explains how environmental features influenced the Random Forest environmental-risk prediction.

SHAP explains model behaviour and should not be interpreted as proof of biological cause and effect.


15. RESEARCH INTERPRETATION
===========================

The final system intentionally distinguishes:

Image AI:
Visible plant condition.

Sensor AI:
Environmental stress or suitability.

Complementary Decision Layer:
Interprets the relationship between both evidence streams.

The image and sensor branches are related but do not represent exactly the same biological target.

Therefore, disagreement between image and sensor outputs can provide useful operational information rather than being treated only as a model error.


16. GITIGNORE RECOMMENDATIONS
=============================

Recommended .gitignore entries:

# Python cache
__pycache__/
*.py[cod]

# Virtual environments
venv/
.venv/

# IDE
.vscode/

# Operating system
.DS_Store
Thumbs.db

Do not commit:

- passwords
- API keys
- Wi-Fi passwords
- private credentials
- unnecessary temporary files
- machine-specific secrets


17. BASIC GIT WORKFLOW
======================

Check current changes:

git status

Stage files:

git add .

Commit:

git commit -m "Update Hydroponic Edge-AI project"

Push to GitHub:

git push origin main


18. TROUBLESHOOTING
===================

18.1 Dashboard Cannot Find Database
-----------------------------------

Check:

ls -l /hydroponic_project/database/hydroponic.db

Also verify that /hydroponic_project points to the correct project directory.


18.2 Streamlit Runs But Another PC Cannot Connect
-------------------------------------------------

Start Streamlit with:

streamlit run /hydroponic_project/dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501

Find the Raspberry Pi IP:

hostname -I

Then open:

http://<PI-IP>:8501


18.3 Model Not Found
--------------------

Check available model files:

find /hydroponic_project/models -type f


18.4 Python Module Error
------------------------

Activate the correct virtual environment:

source venv/bin/activate

Then install the missing dependency.


19. RESEARCH USE
================

This repository was developed as part of an MSc research project at the University of Staffordshire.

The system is a research prototype and should not be considered a replacement for professional agronomic advice or commercial-grade farm-control equipment.


20. AUTHOR
==========

Gabadage Don Tharinda Madushanka
MSc Information Technology Management
University of Staffordshire
Student ID: CB016662
