# Architecture

```
data/*.xlsx ──► data.py ──► PreparedData (scaled, label-encoded, 80:20 stratified split)
                               │
        ┌──────────────────────┼───────────────────────────┐
        ▼                      ▼                           ▼
 models/classical.py     models/stacking.py        models/deep/ (CNN, LSTM, CNN-LSTM,
 (Table 4, Fig. 13)      (Algorithm 1, Table 5)     CNN-BiLSTM, TabNet; Table 6)
        └──────────────────────┬───────────────────────────┘
                               ▼
              training.py / models/deep/trainer.py ──► evaluation.py (metrics, ROC, confusion matrix)
                               │
                               ▼
                persistence.ModelBundle  {model, scaler, label_encoder}
                  │                │                │
                  ▼                ▼                ▼
             inference.py     explain.py (LIME)   edge/service.py ──► ThingsBoard
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Feature list, target, seed (42), split (0.20), dataset location (`$CROP_REC_DATASET` overrides). |
| `data.py` | Load and validate the workbook, drop null/duplicate rows (none exist in the shipped data), Min-Max scale, label-encode, stratified split. |
| `models/classical.py` | One `ModelSpec` per classifier: default (Table 4) and tuned (recorded best parameters) estimators plus the original search space. |
| `models/stacking.py` | One `StackSpec` per Table 5 row: base learners, meta-learner and its search. `svc` is the paper's best (RF + GB + CatBoost → SVC). |
| `models/deep/` | Architectures, registry of final hyper-parameters and Optuna search spaces, training loop, TabNet wrapper. |
| `training.py` | `train_classical`, `search_classical`, `train_stack` returning a `TrainingResult`. |
| `evaluation.py` | Accuracy / weighted precision, recall, F1, inference time, confusion-matrix and ROC figures. |
| `persistence.py` | `ModelBundle`: model + scaler + label encoder in one atomically written `joblib` file; loader for the original three-file layout. |
| `inference.py` | Batch prediction from CSV/XLSX. |
| `explain.py` | LIME explanations in the scaled feature space. |
| `edge/` | Sensor (Modbus), weather (WeatherAPI + BAMIS rainfall), ThingsBoard client, orchestration service, settings. |
| `cli/` | `argparse` sub-commands behind the single `crop-rec` entry point. |
| `reference.py` | Numbers published in the paper, used by `crop-rec reproduce`. |

## Design notes

- **Specs over scripts.** Hyper-parameters live in declarative registries (`MODEL_SPECS`, `STACK_SPECS`,
  `DEEP_SPECS`) copied from the original experiments, so a result can be traced to one place.
- **Context managers for resources.** `ModbusSensor` opens and always closes the serial port; `WeatherClient`
  and `TelemetryClient` own and close an HTTP session; `atomic_path` guarantees readers never see a
  half-written artifact; `timer` and `inference_mode` scope timing and eval mode.
- **Injectable I/O.** Sensor, weather source and telemetry sink are protocols, so the edge pipeline is fully
  testable without hardware or network (`--dry-run` uses the same seams).
- **Secrets.** Never in source: `WEATHER_API_KEY` and `TB_ACCESS_TOKEN` from the environment or flags.
- **Optional dependencies.** PyTorch/TabNet/Optuna (`deep`), LIME (`explain`) and pymodbus/requests (`edge`)
  are extras; importing the core package needs none of them.

## Preprocessing order (as in the paper)

The scaler is fitted on the complete table *before* the split (Section 3.2 describes scaling, then
partitioning), and the same fitted scaler is what the Raspberry Pi loads. Keeping this order is what makes
the deployed scaler consistent with the reported results.

## Security

Model bundles are pickles. Only load files you produced or trust.
