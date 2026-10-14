# Contributing

Thanks for your interest in improving this project.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,explain,edge]"  # add ",deep" for PyTorch / TabNet / Optuna
pre-commit install                    # optional: runs ruff and the notebook cleaner on commit
```

## Checks

```bash
make check        # ruff (lint + format check), mypy --strict, pytest
```

Deep-learning tests are skipped automatically when PyTorch is not installed. Tests marked `deep`
can be selected with `-m deep`.

## Guidelines

- Keep the paper's protocol intact: the dataset, the 80:20 stratified split with seed 42, and the
  recorded hyper-parameters in `models/classical.py`, `models/stacking.py` and
  `models/deep/registry.py` only change together with an explanation of why the published results differ.
- The dataset (`data/New Final DataSet Without Sulphur.xlsx`) must not be modified; a test checks its SHA-256.
- Use `logging` instead of `print` (except for a command's primary output), type-annotate new code and
  add tests for new behaviour. Network, serial-port and filesystem access must be injectable so it can be
  tested without hardware.
- Never commit credentials. The edge service reads `WEATHER_API_KEY` and `TB_ACCESS_TOKEN` from the environment.
- Notebooks in `old/` are an archive; keep their outputs stripped (`python scripts/strip_notebook_outputs.py old/*.ipynb`).

## Pull requests

1. Fork, create a feature branch.
2. Make sure `make check` passes.
3. Describe the change and, if it affects results, include the output of `crop-rec reproduce`.
