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

## ML & DL Training Notebooks

Run the full experimentation pipeline directly in Google Colab or clone locally. Two notebooks are provided:

| Notebook                  | Scope                                                                 | Colab                                                                                                                                                                                                                                                                  |
| ------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CSE499RTK_ML2025.ipynb`  | Classical ML + stacking ensemble (EDA → Optuna tuning → model export) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499RTK_ML2025.ipynb)          |
| `CSE499B_Tabnet.ipynb`    | TabNet deep‑tabular model (sequential attention, sparse masks)        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499B_Tabnet.ipynb)            |
| `CSE499B_CNNBiLSTM.ipynb` | Hybrid CNN → BiLSTM: spatial + bidirectional temporal patterns        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499B_CNNBiLSTM.ipynb)         |
| `CSE499B_CNNLSTM.ipynb`   | CNN → LSTM pipeline (feature maps to sequential learning)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499B_CNNLSTM%20%281%29.ipynb) |
| `CSE499B_cnn.ipynb`       | Compact CNN baseline for tabular‑as‑image reshaping                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/CSE499B_cnn.ipynb)               |

> **Tip:** Colab will prompt you to copy the notebook to your own workspace before executing. A GPU is required for speeding up the training.

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

If you use this work, please cite:

### IEEE
M. S. Sharafat, N. D. Kabya, R. I. Emu, M. U. Ahmed, J. C. Onik, M. A. Islam, and R. Khan, “An IoT-enabled AI system for real-time crop prediction using soil and weather data in precision agriculture,” *Smart Agricultural Technology*, vol. 12, p. 101263, 2025, doi: 10.1016/j.atech.2025.101263.

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
  keywords = {Crop recommendation, Ensemble model, Precision agriculture, Smart irrigation, ThingsBoard cloud},
  abstract = {Context
Precision agriculture leverages advanced technologies such as the Internet of Things (IoT) and artificial intelligence (AI) to enhance crop productivity by providing data-driven insights. In Bangladesh, optimizing crop recommendations using real-time soil and environmental data is crucial for improving agricultural decision-making. However, integrating AI models with IoT devices for instantaneous crop prediction remains a challenge due to computational constraints and the need for model interpretability.
Objective
This study aims to develop an IoT-based crop prediction system that utilizes real-time data on soil nutrients, pH, and weather conditions. The system employs machine learning and deep learning techniques to recommend suitable crops based on local environmental factors. The implementation focuses on deploying the best-performing models on an edge device for real-time predictions as well as ensuring accuracy, efficiency, and accessibility for farmers and agricultural stakeholders.
Methods
The system was developed in Bangladesh using proprietary data from the Soil Resource Development Institute, supported by Habiganj Agricultural University, Sylhet. The dataset consists of 3,300 samples covering 22 crops and eight soil and environmental features. Several machine learning algorithms, including Random Forest, Gradient Boosting, and Stacking ensembles, as well as deep learning models such as TabNet, were evaluated for crop prediction. The best-performing models were deployed on a Raspberry Pi 5 edge device for real-time inference. A weather API was integrated for local humidity and rainfall data, while an RS485 7-in-1 agricultural soil sensor provided real-time measurements of nitrogen (N), phosphorus (P), potassium (K), pH, temperature, and soil moisture. The predictions were displayed on the ThingsBoard IoT platform. Model interpretability was enhanced using the explainable AI technique LIME. A user survey involving farmers, agricultural researchers, and students assessed the usability, accuracy, and reliability of the system.
Results and conclusions
The highest accuracy among machine learning models was achieved using Random Forest (95.8%) and Gradient Boosting (95.5%). The Stacking ensemble technique, with Support Vector Classifier (SVC) as the meta-classifier, achieved the highest overall accuracy of 95.9%. Among deep learning models, TabNet performed best with an accuracy of 92%. The Random Forest model was selected for deployment on the Raspberry Pi due to its lowest inference time and compatibility with Python and TensorFlow Lite. User feedback from the survey provided insights into the system's practical effectiveness and potential areas for improvement. The results demonstrate that integrating AI-driven crop recommendation models with IoT devices can support real-time agricultural decision-making, improving precision farming outcomes.
Significance
This study contributes to precision agriculture by demonstrating an IoT-based crop prediction system that integrates AI-driven recommendations with real-time environmental monitoring. The deployment of the best-performing model on an edge device ensures accessibility and efficiency for users in agricultural settings. By leveraging explainable AI techniques, the study enhances model interpretability, fostering trust and usability among farmers and agricultural researchers. The findings highlight the potential of AI and IoT in improving crop selection, optimizing resource usage, and supporting sustainable agricultural practices in Bangladesh and beyond. The implementation code and private dataset are available at: https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data.}
}
```

## Acknowledgements

* **Habiganj Agricultural University Research System (HAURES)** – funding & field support.
* **Soil Resource Development Institute (SRDI)** – soil data access.
* **Bangladesh Agro‑Meteorological Information Service (BAMIS)** – weather APIs.
* Southeast Bank PLC – equipment sponsorship.

---

<p align="center">🍀 Smart Farming and Precision Agriculture in Bangladesh: A Transformative Initiative by MD Shaifullah Sharafat, accompanied by Rahimul Islam, Nilavro Das Kabya
 🍀</p>
