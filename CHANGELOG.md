# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0]

### Added
- Installable `crop_recommendation` package (`src` layout) with the `crop-rec` command:
  `train`, `evaluate`, `predict`, `explain`, `edge` and `reproduce`.
- Typed, documented modules for data preparation, classical and stacking models, PyTorch / TabNet
  models, evaluation, LIME explanations and the Raspberry Pi service.
- Test suite (including checks that the dataset is byte-identical and that the deep architectures have
  the exact parameter counts reported in the paper), CI, Dockerfile, pre-commit, Makefile.
- `docs/` with architecture notes, a results-reproduction report and edge-deployment instructions.

### Changed
- The Raspberry Pi script is now `crop_recommendation.edge`: serial port, HTTP session and sensor are
  context-managed; credentials come from environment variables instead of source code; the
  stabilisation delay defaults to the one minute described in the paper (the original script waited 6 s).
- Original notebooks moved to `old/` with cell outputs removed.
- Dataset moved to `data/` (content unchanged).

### Removed
- Descriptions of files that were never part of the repository (configs, `requirements*.txt`,
  git-LFS model weights, etc.) were removed from the README.
