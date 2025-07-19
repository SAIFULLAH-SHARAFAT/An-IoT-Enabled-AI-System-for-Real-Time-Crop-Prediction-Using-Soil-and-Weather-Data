# IoT‑Enabled AI System for Real‑Time Crop Prediction

> **Precision agriculture meets edge AI – real‑time crop recommendations from soil & weather data on a Raspberry Pi 5**
---

## Table of Contents

1. [Features](#features)
2. [Hardware Overview](#hardware-overview)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
5. [Model Training & Evaluation](#model-training--evaluation)
6. [Edge Deployment on Raspberry Pi 5](#edge-deployment-on-raspberry-pi-5)
7. [ThingsBoard Dashboard](#thingsboard-dashboard)
8. [Explainable AI](#explainable-ai)
9. [Results](#results)
10. [Limitations & Road‑Map](#limitations--road‑map)
11. [Contributing](#contributing)
12. [License](#license)
13. [Citation](#citation)
14. [Acknowledgements](#acknowledgements)

---

## Features

* **Real‑time sensor fusion** – Combines soil macro‑nutrients (N, P, K), pH, temperature & moisture from an RS‑485 7‑in‑1 probe with live weather API feeds (humidity & rainfall).
* **State‑of‑the‑art ML ⇄ DL pipeline** – Supports Random Forest, Gradient Boosting & an **SVC‑based stacking ensemble** (95.9 % test accuracy) plus TabNet, CNN & LSTM variants.
* **Edge ready** – Optimised `.pkl` models quantised via TensorFlow‑Lite for < 70 ms inference on **Raspberry Pi 5**.
* **Plug‑and‑play dashboard** – Auto‑publishes time‑series data & crop recommendations to a ThingsBoard cloud instance with alarms & mobile access.
* **Explainable AI** – Built‑in LIME wrappers so agronomists can inspect feature contributions crop‑by‑crop.
* **Survey‑driven UX** – Early farmer feedback (n = 10) used to refine interface clarity & responsiveness.

---

## Hardware Overview

| Component                              | Purpose                                     | Approx. Cost |
| -------------------------------------- | ------------------------------------------- | ------------ |
| **Raspberry Pi 5** (8 GB)              | Edge inference, API calls, MQTT/HTTP client | 14 500 BDT   |
| **7‑in‑1 Soil Sensor** (RS‑485, IP‑68) | N, P, K, pH, temp, moisture                 | 16 500 BDT   |
| USB‑to‑RS‑485 Converter                | Modbus RTU ↔ USB bridge                     | 1 000 BDT    |
| 20 000 mAh Power Bank                  | † Optional field power                      | 1 500 BDT    |
| Misc. enclosure + cooler               | Thermal & weather protection                | 3 000 BDT    |

† Runs Pi 5 + sensor stack for ≈ 8–9 h.

A wiring schematic is available in [`docs/hardware_schematic.pdf`](docs/hardware_schematic.pdf).

---

## Project Structure

```text
├── data/                  # (private) original & processed CSVs
├── notebooks/             # Jupyter EDA, LIME, Optuna trials
├── src/
│   ├── train.py           # generic trainer (scikit‑learn & PyTorch)
│   ├── infer.py           # batch inference script
│   ├── edge/              # Raspberry Pi 5 specific code
│   └── utils/             # helpers – transforms, metrics, plots
├── iot/
│   ├── thingsboard_rules/ # JSON exports for TB widgets & alarms
│   └── modbus_reader.py   # sensor polling service (pymodbus)
├── requirements.txt
├── Dockerfile             # CPU reproducible env (Ubuntu 22.04)
├── LICENSE                # Apache 2.0
└── README.md
```

---

## Dataset

The Habiganj soil‑weather dataset used in our paper is publicly hosted:

* **GitHub (XLSX)** – [New Final DataSet Without Sulphur.xlsx](https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/New%20Final%20DataSet%20Without%20Sulphur.xlsx)
  Rows × Cols = 3 300 × 9 (8 features + `crop` label)
  License = Apache 2.0 (same as repo)

## ML Training Notebook

A Google Colab‑ready notebook walks through the full **machine‑learning pipeline**—from EDA and preprocessing to Optuna hyper‑parameter search, evaluation, and model export:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499RTK_ML2025.ipynb)

* **Filename**: `CSE499RTK_ML2025.ipynb`
* **Outputs**: `models/best_rf.pkl`, metric logs, ROC & confusion‑matrix plots under `reports/`.

> **Local run**: `jupyter lab notebooks/CSE499RTK_ML2025.ipynb --NotebookApp.token=''`

---

## Getting Started

### Prerequisites

* Python ≥ 3.10 (tested on 3.11)
* pip / virtualenv **or** conda
* Git + Git LFS (for large model weights)
* Docker (optional, for zero‑setup runs)

### Installation

```bash
# 1) Clone
$ git clone https://github.com/<your‑org>/iot‑crop‑recommendation.git
$ cd iot‑crop‑recommendation

# 2) Create & activate venv
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install Python deps
$ pip install -r requirements.txt

# 4) (Optional) Pull pretrained model (~3 MB)
$ git lfs pull --include "models/best_rf.pkl"
```

### Quick Start – Local Inference

```bash
$ python src/infer.py \
      --model_path models/best_rf.pkl \
      --input_csv examples/sample_sensor_readings.csv
```

### Docker

```bash
$ docker build -t crop‑rec .
$ docker run --rm -it crop‑rec python src/infer.py --help
```

---

## Model Training & Evaluation

1. **Pre‑process**

   ```bash
   python src/train.py prep --cfg configs/default.yaml
   ```
2. **Hyper‑parameter tuning** (Optuna)

   ```bash
   python src/train.py tune --study_name rf_habiganj --n_trials 100
   ```
3. **Train best model & export**

   ```bash
   python src/train.py fit --cfg configs/rf_best.yaml
   ```

   The script outputs metrics (accuracy, F1‑macro, ROC) and saves the model under `models/`.

All scores in the paper are reproduced via `scripts/reproduce_metrics.sh`.

---

## Edge Deployment on Raspberry Pi 5

```bash
# On Pi 5 (64‑bit Raspberry Pi OS)
$ sudo apt update && sudo apt install python3-pip git
$ git clone https://github.com/<your‑org>/iot‑crop‑recommendation.git
$ cd iot‑crop‑recommendation
$ pip install -r requirements_edge.txt

# Connect RS‑485 → USB & verify tty
$ ls /dev/ttyUSB*

# Start the Modbus polling + inference + ThingsBoard uploader
$ python iot/modbus_reader.py --tty /dev/ttyUSB0 \
                              --model models/best_rf.pkl \
                              --tb_token <THINGBOARD_DEVICE_TOKEN>
```

The service publishes JSON payloads every 60 s to `/api/v1/<token>/telemetry`.

---

## ThingsBoard Dashboard

Import widgets via **Dashboards → Import** with `iot/thingsboard_rules/dashboard_export.json`.

Key widgets:

* Real‑time nutrient gauges (N, P, K, pH, moisture)
* Weather cards (humidity, rainfall)
* Suggested crop (string widget)
* Alarms for extreme pH / moisture

---

## Explainable AI

Run LIME on a single prediction:

```bash
python notebooks/lime_demo.py --model models/best_rf.pkl --n_samples 500
```

Generates an interactive plot (HTML) under `reports/lime/`.

---

## Results

| Model             | Accuracy   | Inference (CPU, Pi 5) |
| ----------------- | ---------- | --------------------- |
| **Random Forest** | **95.8 %** | **60.8 ms**           |
| Gradient Boosting | 95.5 %     | 69.3 ms               |
| Stacked SVC       | 95.9 %     | 435 ms                |
| TabNet (DL)       | 92.0 %     | 1.2 s                 |

Full metrics in `reports/`.

---

## Limitations & Road‑Map

* **Regional bias** – Current dataset limited to Habiganj district; broader agro‑ecological coverage planned (2025 Q4).
* **Micro‑nutrients & pests** – Future sensor integration for S, Zn & real‑time disease detection.
* **Federated learning** – Edge‑to‑cloud update pipeline under development to respect data privacy.

See the [open issues](https://github.com/<your‑org>/iot‑crop‑recommendation/issues) for full to‑do list.

---

## Contributing

Contributions are welcome 🙌

1. Fork → feature branch → PR.
2. Follow `black` + `isort` + `ruff` formatting (`pre‑commit` hooks included).
3. Add/adjust unit tests (`pytest`).
4. Ensure CI passes.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

---

## License

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
```

See [`LICENSE`](LICENSE) the full text.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{sharafat2025iotcrop,
  title   = {An IoT‑Enabled AI System for Real‑Time Crop Prediction Using Soil and Weather Data in Precision Agriculture},
  author  = {Sharafat, Md Shaifullah and Kabya, Nilavro Das and Islam, Rahimul and Ahmed, Mehrab Uddin and Onik, Jakaria Chowdhury and Islam, Mohammad Aminul and Khan, Riasat},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2025},
  note    = {Early Access}
}
```

---

## Acknowledgements

* **Habiganj Agricultural University Research System (HAURES)** – funding & field support.
* **Soil Resource Development Institute (SRDI)** – soil data access.
* **Bangladesh Agro‑Meteorological Information Service (BAMIS)** – weather APIs.
* Southeast Bank PLC – equipment sponsorship.

---

<p align="center">🍀 Smart Farming and Precision Agriculture in Bangladesh: A Transformative Initiative by Md. Shaifullah Sharafat, accompanied by Rahimul Islam, Nilavro Das Kabya
 🍀</p>
