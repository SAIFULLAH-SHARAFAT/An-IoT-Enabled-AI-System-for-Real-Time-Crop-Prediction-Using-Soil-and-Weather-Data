# IoT-Enabled AI System for Real-Time Crop Prediction

[![CI](https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.atech.2025.101263-orange.svg)](https://doi.org/10.1016/j.atech.2025.101263)

> **Precision agriculture meets edge AI: real-time crop recommendations from soil and weather data on a Raspberry Pi 5.**

Official implementation of the paper *"An IoT-enabled AI system for real-time crop prediction using soil and
weather data in precision agriculture"* (Smart Agricultural Technology, 2025). A 7-in-1 RS-485 soil sensor and a
weather API feed a machine-learning model running on a Raspberry Pi 5; recommendations are published to a
ThingsBoard dashboard.

## Table of contents

1. [Features](#features)
2. [Hardware overview](#hardware-overview)
3. [Project structure](#project-structure)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Edge deployment on Raspberry Pi 5](#edge-deployment-on-raspberry-pi-5)
9. [ThingsBoard dashboard](#thingsboard-dashboard)
10. [Explainable AI](#explainable-ai)
11. [Limitations and road-map](#limitations-and-road-map)
12. [Development](#development)
13. [License](#license)
14. [Citation](#citation)
15. [Acknowledgements](#acknowledgements)

## Features

- **Real-time sensor fusion**: soil N, P, K, pH, temperature and moisture from an RS-485 7-in-1 probe combined with
  live humidity from WeatherAPI and monthly rainfall from BAMIS.
- **ML and DL pipelines**: Random Forest, Gradient Boosting, a **stacking ensemble with an SVC meta-learner
  (95.9 % test accuracy)**, plus TabNet, CNN, LSTM, CNN-LSTM and CNN-BiLSTM.
- **Edge ready**: the Random Forest is deployed on a Raspberry Pi 5 for its low inference time (0.0608 s in the paper).
- **Dashboard**: crop recommendations and sensor time series are posted to ThingsBoard over HTTP.
- **Explainable AI**: LIME explanations of individual predictions.
- **User survey**: early feedback from 10 farmers, researchers and students informed the interface.
- **One command-line tool** (`crop-rec`) for training, evaluation, prediction, explanation, reproduction of the
  paper's tables and running the edge service.

## Hardware overview

| Component | Purpose | Approx. cost |
| --- | --- | --- |
| Raspberry Pi 5 (8 GB) | Edge inference, API calls, sensor data storage | 14 500 BDT (about 20 000 BDT with case and cooler) |
| 7-in-1 soil sensor (RS-485, IP68) with Modbus converter and accessories | N, P, K, pH, temperature, moisture | 16 500 BDT |
| 20 000 mAh power bank (optional) | Field power, about 8-9 h of operation | 1 500 BDT |

Total prototype cost reported in the paper: **36 500 BDT (about USD 297)**, excluding the power bank.

## Project structure

```text
.
├── src/crop_recommendation/     # installable package
│   ├── config.py  data.py  training.py  evaluation.py  persistence.py  inference.py  explain.py
│   ├── models/                  # classical.py, stacking.py, deep/ (architectures, registry, trainer, tuning, tabnet)
│   ├── edge/                    # sensor.py (Modbus), weather.py, thingsboard.py, service.py, settings.py
│   └── cli/                     # argparse sub-commands behind `crop-rec`
├── tests/                       # pytest suite
├── data/                        # New Final DataSet Without Sulphur.xlsx
├── docs/                        # architecture, reproducing results, edge deployment
├── old/                         # archived original notebooks (outputs removed) and the original Pi script
├── scripts/                     # maintenance utilities
├── pyproject.toml  Dockerfile  Makefile  CITATION.cff  CHANGELOG.md  CONTRIBUTING.md  LICENSE
```

## Dataset

The Habiganj soil-weather dataset used in the paper: **3 300 samples, 22 crops (150 each), 8 features** collected
from eight unions of Habiganj District, Sylhet, Bangladesh. Soil properties come from the Soil Resource Development
Institute (SRDI); temperature, humidity and rainfall come from the Bangladesh Agro-Meteorological Information
Service (BAMIS).

| Feature | Unit |
| --- | --- |
| `N`, `P`, `K` | soil nutrients |
| `pH` | soil pH |
| `Temp(°C)` | temperature |
| `Humidity(%)` | relative humidity |
| `Moisture(%)` | soil moisture |
| `Rainfall(cm)` | rainfall |
| `Crop` | label (22 classes) |

File: [`data/New Final DataSet Without Sulphur.xlsx`](data/New%20Final%20DataSet%20Without%20Sulphur.xlsx)
(Apache 2.0, same as this repository). The file is never modified by the code, and a test verifies its checksum.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data.git
cd An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[explain,edge]"       # core + LIME + Raspberry Pi dependencies
pip install -e ".[deep]"               # optional: PyTorch, TabNet, Optuna
pip install -e ".[dev]"                # optional: tests and linters
```

Docker (CPU): `docker build -t crop-rec .` then `docker run --rm crop-rec train classical --model rf`.

## Usage

```bash
crop-rec --help
```

| Command | What it does |
| --- | --- |
| `crop-rec train classical` | Train a classifier (`--model dt rf nb lr svc knn gb xgb catboost all`; `--variant default tuned search`). |
| `crop-rec train stack` | Stacking ensemble, RF + GB + CatBoost base learners (`--meta svc` is the paper's best). |
| `crop-rec train deep` | CNN, LSTM, CNN-LSTM, CNN-BiLSTM or TabNet (`--tune` runs the Optuna search). |
| `crop-rec evaluate` | Metrics, confusion matrix and ROC curves of a saved model on the 20 % test split. |
| `crop-rec predict` | Crop recommendations for a CSV/XLSX of raw sensor readings. |
| `crop-rec explain` | LIME explanation (HTML and PNG) for one test-set prediction. |
| `crop-rec edge` | Run the Raspberry Pi service (sensor, weather, prediction, ThingsBoard). |
| `crop-rec reproduce` | Re-run the paper's tables and print them next to the published numbers. |

Typical session:

```bash
crop-rec train classical --model rf --variant tuned          # -> models/rf-tuned/model.joblib
crop-rec evaluate --model-path models/rf-tuned/model.joblib  # -> reports/evaluation/
crop-rec predict  --model-path models/rf-tuned/model.joblib --input readings.csv
crop-rec explain  --model-path models/rf-tuned/model.joblib --index 5
crop-rec train stack --meta svc                              # the 95.9 % configuration (several minutes)
crop-rec reproduce --only table4 table6
```

`readings.csv` needs the columns `N, P, K, pH, Temp(°C), Humidity(%), Moisture(%), Rainfall(cm)` with raw
(unscaled) values. Every command accepts `--seed` (default 42) and `--data` to use another workbook; the dataset path
can also be set with `CROP_REC_DATASET`.

Protocol (paper Section 3): duplicates and nulls removed, Min-Max scaling, label encoding, stratified 80:20
train/test split, seed 42, accuracy / precision / recall / F1.

## Results

Test accuracy reported in the paper (3 300 samples, 20 % hold-out):

| Model | Accuracy | Inference time on the Pi 5 |
| --- | ---: | ---: |
| Random Forest (deployed) | **95.8 %** | **0.0608 s** |
| Gradient Boosting | 95.5 % | 0.0693 s |
| Stacking, SVC meta-learner (RF + GB + CatBoost) | **95.9 %** | 0.4347 s |
| TabNet (best deep model) | 92.0 % | not reported |
| LSTM / CNN / CNN-LSTM / CNN-BiLSTM | 91.8 / 89.6 / 88.9 / 86.8 % | not reported |

`crop-rec reproduce` re-runs these experiments from this repository. A side-by-side comparison, together with
notes on where and why the re-run differs from the published tables, is in
[`docs/reproducing-results.md`](docs/reproducing-results.md).

## Edge deployment on Raspberry Pi 5

```bash
export WEATHER_API_KEY=...        # WeatherAPI key
export TB_ACCESS_TOKEN=...        # ThingsBoard device access token
crop-rec edge --model-path models/rf-tuned/model.joblib --port /dev/ttyUSB0
```

Without hardware, `crop-rec edge --model-path ... --dry-run --startup-delay 0` runs the full pipeline with a
simulated sensor and logs the payload instead of uploading it. Full instructions (wiring, options, payload,
using the original `.pkl` files) are in [`docs/edge-deployment.md`](docs/edge-deployment.md).

## ThingsBoard dashboard

Telemetry is posted to `/api/v1/<token>/telemetry` with the keys `N, P, K, Moisture, Temperature, pH,
Humidity, Rainfall, RecommendedCrop, EC, ts`. The prototype dashboard shows:

- real-time nutrient gauges (N, P, K, pH, moisture),
- weather cards (humidity, rainfall),
- the suggested crop (string widget),
- alarms for extreme pH, high soil temperature or low moisture.

## Explainable AI

```bash
crop-rec explain --model-path models/rf-tuned/model.joblib --index 5 --output-dir reports/lime
```

Writes an interactive HTML report and a PNG showing the feature contributions of the selected prediction
(features are in the Min-Max scaled space used for training).

## Limitations and road-map

- **Regional bias**: the dataset covers Habiganj District only; broader agro-ecological coverage is planned.
- **Micro-nutrients and pests**: future sensor integration for sulphur and zinc and real-time disease detection.
- **Coarse weather inputs**: BAMIS monthly averages and API feeds do not capture intra-seasonal anomalies.
- **Federated learning**: an edge-to-cloud update pipeline is under development to respect data privacy.

## Development

```bash
pip install -e ".[dev,explain,edge,deep]"
make check          # ruff, mypy --strict, pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](docs/architecture.md). The original research
notebooks are archived in [`old/`](old/README.md).

## License

Released under the [Apache License 2.0](LICENSE).

## Citation

If you use this work, please cite:

### IEEE

```ieee
M. S. Sharafat, N. D. Kabya, R. I. Emu, M. U. Ahmed, J. C. Onik, M. A. Islam, and R. Khan, “An IoT-enabled AI system for real-time crop prediction using soil and weather data in precision agriculture,” *Smart Agricultural Technology*, vol. 12, p. 101263, 2025, doi: 10.1016/j.atech.2025.101263.
```

### BibTeX

```bibtex
@article{SHARAFAT2025101263,
  title = {An IoT-enabled AI system for real-time crop prediction using soil and weather data in precision agriculture},
  journal = {Smart Agricultural Technology},
  volume = {12},
  pages = {101263},
  year = {2025},
  issn = {2772-3755},
  doi = {https://doi.org/10.1016/j.atech.2025.101263},
  url = {https://www.sciencedirect.com/science/article/pii/S2772375525004940},
  author = {MD Shaifullah Sharafat and Nilavro Das Kabya and Rahimul Islam Emu and Mehrab Uddin Ahmed and Jakaria Chowdhury Onik and Mohammad Aminul Islam and Riasat Khan},
  keywords = {Crop recommendation, Ensemble model, Precision agriculture, Smart irrigation, ThingsBoard cloud}
}
```

## Acknowledgements

- **Habiganj Agricultural University Research System (HAURES)**: funding and field support.
- **Soil Resource Development Institute (SRDI)**: soil data access.
- **Bangladesh Agro-Meteorological Information Service (BAMIS)**: weather data.
- Southeast Bank PLC: equipment sponsorship.

<p align="center">Smart Farming and Precision Agriculture in Bangladesh: A Transformative Initiative by MD Shaifullah Sharafat, accompanied by Rahimul Islam and Nilavro Das Kabya</p>
