# IoT‑Enabled AI System for Real‑Time Crop Prediction

> **Precision agriculture meets edge AI – real‑time crop recommendations from soil & weather data on a Raspberry Pi 5**
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Hardware Architecture](#hardware-architecture)
4. [Dataset](#dataset)
5. [Model Zoo & Results](#model-zoo--results)
6. [Quick Start Guide](#quick-start-guide)
7. [Deploy to Raspberry Pi 5](#deploy-to-raspberry-pi-5)
8. [Dashboard & API](#dashboard--api)
9. [Explainable AI](#explainable-ai)
10. [Repository Structure](#repository-structure)
11. [Roadmap](#roadmap)
12. [Citation](#citation)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository accompanies the manuscript:

> **“An IoT‑Enabled AI System for Real‑Time Crop Prediction Using Soil and Weather Data in Precision Agriculture.”**
> *MD Shaifullah Sharafat et al., 2025.*

We present an **edge‑friendly crop recommendation pipeline** that fuses real‑time soil sensor readings and micro‑weather feeds to advise farmers on the best crop to plant **in seconds**, even in low‑connectivity fields.

---

## Key Features

* **Plug‑and‑play hardware**: Raspberry Pi 5 + RS‑485 7‑in‑1 soil sensor (N, P, K, pH, temperature, moisture).
* **ML & DL ensemble** with AutoML tuning (Optuna) – best trade‑off model (Random Forest) → **95.8 % accuracy**, **60 ms inference** on‑device.
* **Real‑time dashboard** on [ThingsBoard](https://thingsboard.io/) – live charts, alarms, and crop suggestions.
* **Explainable AI** via LIME to boost farmers’ trust.
* **Modular codebase** – train locally, export to TensorFlow Lite / `joblib`, deploy on Pi with a single script.
* **Docker‑ready** dev environment.

---

## Hardware Architecture

```
┌────────────────────┐      RS‑485      ┌─────────────────┐
│ 7‑in‑1 Soil Sensor │◀───────────────▶│ USB ↔ RS‑485     │
└────────────────────┘                └─────────────────┘
                                          │ USB
┌────────────────────────────────────────────────────────┐
│                 Raspberry Pi 5 (8 GB)                 │
│ ─ Flask API & MQTT publisher                          │
│ ─ Pre‑trained RF model (TFLite / joblib)              │
│ ─ Weather API fetcher (humidity, rainfall)            │
└────────────────────────────────────────────────────────┘
           │ HTTPS                                          
           ▼                                               
┌────────────────────────────────────────────────────────┐
│                 ThingsBoard Cloud                      │
│   Live dashboard · Alerts · Historical analytics       │
└────────────────────────────────────────────────────────┘
```

*Bill of materials ≈ 36 500 BDT (≈ \$297), incl. power bank for 8–9 h field runtime.*

---

## Dataset

| Source                      | Samples | Classes  | Features                                             |
| --------------------------- | ------- | -------- | ---------------------------------------------------- |
| SRDI + BAMIS (Habiganj, BD) | 3,300   | 22 crops | N, P, K, pH, soil temp, moisture, rainfall, humidity |

The Habiganj soil‑weather dataset used in our paper is here:

* **GitHub (XLSX)** – [New Final DataSet Without Sulphur.xlsx](https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/blob/main/New%20Final%20DataSet%20Without%20Sulphur.xlsx)
* **Rows × Cols** – 3 300 × 9 (8 features + `crop` label)
* **License** – Apache 2.0 (same as this repo)

Download via CLI:

```bash
wget -O data/habiganj_dataset.xlsx "https://raw.githubusercontent.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data/main/New%20Final%20DataSet%20Without%20Sulphur.xlsx"
```

After placing the file under `data/`, proceed with preprocessing & training as described below.

---


## Model Zoo & Results

| Category    | Model               | Accuracy   | F1‑macro | Inference (Pi 5) |
| ----------- | ------------------- | ---------- | -------- | ---------------- |
| ML          | Random Forest       | **95.8 %** | 0.958    | **0.06 s**       |
| ML          | Gradient Boosting   | 95.5 %     | 0.955    | 0.07 s           |
| ML Ensemble | Stacking (SVC meta) | 95.9 %     | 0.959    | 0.43 s           |
| DL          | TabNet              | 92.0 %     | 0.920    | 1.2 s            |
| DL          | CNN‑BiLSTM          | 86.8 %     | 0.867    | 1.5 s            |

See `notebooks/` for full experiments, ROC curves, and Optuna studies.

---

## Quick Start Guide

### 1. Clone & set up

```bash
# clone repo
$ git clone https://github.com/<your‑user>/<repo>.git
$ cd <repo>

# create environment
$ conda env create -f env.yml  # or: pip install -r requirements.txt
$ conda activate crop‑iot
```

### 2. Train locally (optional)

```bash
$ python src/train.py \
    --config configs/rf_default.yaml \
    --data data/sample_dataset.csv
# models saved in models/
```

### 3. Launch dashboard (local dev)

```bash
$ docker compose up thingsboard
```

---

## Deploy to Raspberry Pi 5

1. Flash **Raspberry Pi OS 64‑bit** to a micro‑SD (SDR104 recommended).
2. Enable SSH & Wi‑Fi; install dependencies:

   ```bash
   $ sudo apt update && sudo apt install python3‑pip python3‑venv
   $ python3 -m venv ~/crop‑env && source ~/crop‑env/bin/activate
   $ pip install -r pi/requirements_pi.txt
   ```
3. Plug the **USB ↔ RS‑485** adapter and sensor, note `/dev/ttyUSB0`.
4. Copy the trained model & scaler:

   ```bash
   $ scp models/rf_best.pkl pi@<pi‑ip>:~/crop‑project/models/
   ```
5. Run the edge script:

   ```bash
   $ python pi/edge_app.py --port /dev/ttyUSB0 --tb‑token <TB_DEVICE_TOKEN>
   ```
6. Open **ThingsBoard** → Devices → *Pi‑Field‑001* → **Latest telemetry**.

---

## Dashboard & API

| Endpoint   | Method | Description                                                                      |
| ---------- | ------ | -------------------------------------------------------------------------------- |
| `/predict` | POST   | JSON `{N,P,K,pH,temp,moisture,humidity,rainfall}` → returns `{crop:"Boro Rice"}` |
| `/sensor`  | GET    | Latest raw soil readings                                                         |
| `/health`  | GET    | Service heartbeat                                                                |

Live demo screenshots are in `docs/dashboard/`.

---

## Explainable AI

Run `notebooks/05_LIME_explainer.ipynb` to generate local surrogate explanations and feature contribution plots *per prediction* (supports both scikit‑learn & TFLite models).

---

## Repository Structure

```
├── data/                 # toy dataset & scripts to prep full dataset
├── docs/                 # figures, architecture diagrams, paper draft
├── notebooks/            # Jupyter explorations & Optuna studies
├── src/
│   ├── data_utils.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── pi/                   # edge‑device code (Flask API + Modbus reader)
├── env.yml               # conda environment (dev)
└── LICENSE
```

---

## Roadmap

* [ ] Add **weather‑aware temporal LSTM** to capture seasonality.
* [ ] Integrate **federated learning** for privacy‑preserving model updates.
* [ ] Support **LoRaWAN** communication for off‑grid farms.
* [ ] Extend dataset beyond Sylhet to national coverage (+ micro‑nutrient sensors).

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Sharafat2025CropIoT,
  author    = {MD Shaifullah Sharafat, Nilavro Das Kabya, Rahimul Islam Emu, Mehrab Uddin Ahmed, Jakaria Chowdhury Onik, Mohammad Aminul Islam and Riasat Khan},
  title     = {An IoT‑Enabled AI System for Real‑Time Crop Prediction Using Soil and Weather Data in Precision Agriculture},
  journal   = {Smart Agricultural Technology},
  year      = {2025},
  note      = {In review}
}
```

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss your proposal.
Make sure to follow the style guidelines in `CONTRIBUTING.md` and run `pre‑commit` before pushing.

---

## License

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/
```

See [`LICENSE`](LICENSE) for the full text.

---

## Acknowledgements

* Habiganj Agricultural University Research System (HAURES) & Southeast Bank PLC for funding.
* Soil Resource Development Institute (SRDI) & BAMIS for data access.
* Community contributors for suggestions and testing.

<p align="center">🍀 Happy Farming! 🍀</p>
