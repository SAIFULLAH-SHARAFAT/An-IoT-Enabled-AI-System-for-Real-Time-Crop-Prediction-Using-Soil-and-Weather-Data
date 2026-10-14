# Archive

Original research artifacts kept for provenance. They are **not** maintained; use the `crop-rec`
command-line tool (see the top-level README) for reproducible runs.

| File | Content |
| --- | --- |
| `CSE499RTK_ML2025.ipynb` | EDA, classical models, hyper-parameter tuning, stacking, LIME/SHAP |
| `CSE499B_Tabnet.ipynb` | TabNet |
| `CSE499B_cnn.ipynb` | CNN |
| `CSE499B_CNNLSTM.ipynb` | CNN-LSTM |
| `CSE499B_CNNBiLSTM.ipynb` | CNN-BiLSTM |
| `Crop_Recommendation(rasp_file).py` | Raspberry Pi script used for the prototype (superseded by `crop_recommendation.edge`) |

Cell outputs were removed to keep the repository small (`scripts/strip_notebook_outputs.py`); sources are
unchanged. The notebooks read `New Final DataSet Without Sulphur.xlsx`, which now lives in `../data/`.

Cell execution order in `CSE499RTK_ML2025.ipynb` was not linear, so several cells reuse variables from
earlier or later cells. See `../docs/reproducing-results.md` for what this means for the published tables.
