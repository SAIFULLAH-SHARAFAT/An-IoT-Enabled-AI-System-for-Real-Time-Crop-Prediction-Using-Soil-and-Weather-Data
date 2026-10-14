# Edge deployment on Raspberry Pi 5

## Hardware (paper Section 3.5)

| Component | Purpose |
| --- | --- |
| Raspberry Pi 5 (8 GB) | Inference, API calls, HTTP client |
| RS-485 7-in-1 soil sensor (N, P, K, pH, temperature, moisture, EC) | Soil measurements over Modbus-RTU |
| USB-to-RS-485 converter | Connects the sensor to the Pi |
| 20 000 mAh power bank (optional) | Roughly 8-9 h of field operation |

Serial settings used by the prototype: 4800 baud, 8 data bits, no parity, 1 stop bit, Modbus unit id 1.

## Install

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone https://github.com/SAIFULLAH-SHARAFAT/An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data.git
cd An-IoT-Enabled-AI-System-for-Real-Time-Crop-Prediction-Using-Soil-and-Weather-Data
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[edge]"
```

Train (or copy) the model bundle, for example on a workstation:

```bash
crop-rec train classical --model rf --variant search --output-dir models
```

## Run

```bash
ls /dev/ttyUSB*                                    # confirm the converter
export WEATHER_API_KEY=...                          # weatherapi.com key
export TB_ACCESS_TOKEN=...                          # ThingsBoard device access token
crop-rec edge --model-path models/rf-search/model.joblib --port /dev/ttyUSB0
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--startup-delay` | `60` | Seconds to let the sensor stabilise (paper: one minute). |
| `--interval` | `0` | `0` = one cycle then exit; otherwise repeat every N seconds. |
| `--location` | `Dhaka` | WeatherAPI location query. |
| `--csv-log PATH` | off | Also append every cycle, with timestamp, to a CSV (offline record). |
| `--dry-run` | off | Simulated sensor/weather, nothing uploaded; needs no hardware, network or secrets. |

Try the whole pipeline without hardware: `crop-rec edge --model-path models/rf-tuned/model.joblib --dry-run --startup-delay 0`.

### Using the three original `.pkl` files

```bash
crop-rec edge --model-path random_search_model.pkl --scaler-path scaler.pkl --encoder-path label_encoder.pkl
```

## Data flow (one cycle)

1. Fetch current humidity from WeatherAPI (skip the cycle if unavailable).
2. Read holding registers 0-6 and decode: moisture ÷ 10, temperature ÷ 10, EC (raw, µS/cm), pH ÷ 10, N × 2, P × 2, K × 2.
3. Rainfall = BAMIS average for the current calendar month (`edge/weather.py`).
4. Build the eight model features (EC is sent to the dashboard but is not a model input), scale, predict, decode the crop name.
5. POST JSON to `<host>/api/v1/<token>/telemetry` with keys `N, P, K, Moisture, Temperature, pH, Humidity, Rainfall, RecommendedCrop, EC, ts`.

## Notes

- The telemetry payload is identical to the one produced by the deployed prototype so existing ThingsBoard widgets keep working.
- The monthly rainfall table is carried over verbatim from the prototype. It holds BAMIS averages for Dhaka, where the
  prototype was tested (`--location Dhaka`), whereas the training data was collected in Habiganj District, so its values
  (9.4-819.2) are larger than the `Rainfall(cm)` column of the dataset (40-250). For a deployment elsewhere, replace the
  table in `edge/weather.py` with figures for that location.
- Weather requests use HTTPS and have timeouts; failures skip the cycle instead of crashing the service.
