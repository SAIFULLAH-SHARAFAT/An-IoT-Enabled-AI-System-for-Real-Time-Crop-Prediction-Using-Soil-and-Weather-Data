# Reproducing the paper's results

```bash
crop-rec reproduce                      # Tables 4, 5 (SVC meta-learner only), 6 and Fig. 13
crop-rec reproduce --all-meta           # every meta-learner of Table 5 (slow)
crop-rec reproduce --only table4 fig13  # a subset
```

The command writes `reports/reproduction/comparison.md` and `comparison.json`.

**Protocol (paper Section 3.2):** null/duplicate removal, Min-Max scaling, label encoding, 80:20 stratified split,
`random_state=42`. Hyper-parameters are the ones recorded in the original notebooks.

## Results of `crop-rec reproduce --all-meta` (seed 42, scikit-learn 1.9, CPU)

Percentages are test accuracy (Table 5 CV rows: best cross-validation accuracy). "Delta" is reproduced minus paper, in percentage points.

| Section | Item | Paper | Reproduced | Delta (pt) | Note |
|---|---|---:|---:|---:|---|
| Table 4 (default) | GaussianNB | 92.6 | 92.6 | -0.0 |  |
| Table 4 (default) | KNN | 88.0 | 88.0 | +0.0 |  |
| Table 4 (default) | Decision Tree | 93.0 | 93.0 | +0.0 |  |
| Table 4 (default) | Random Forest | 95.8 | 95.8 | -0.0 |  |
| Table 4 (default) | Logistic Regression | 83.6 | 83.6 | +0.0 |  |
| Table 4 (default) | SVC | 90.5 | 90.5 | -0.0 |  |
| Table 4 (default) | AdaBoost | 10.0 | 10.0 | +0.0 |  |
| Table 4 (default) | Gradient Boosting | 95.5 | 95.5 | -0.0 |  |
| Table 4 (default) | CatBoost | 92.6 | 95.2 | +2.6 |  |
| Fig. 13 (tuned) | Decision Tree | 93.8 | 93.8 | -0.0 |  |
| Fig. 13 (tuned) | GaussianNB | 92.6 | 92.6 | -0.0 |  |
| Fig. 13 (tuned) | Logistic Regression | 91.1 | 90.5 | -0.6 |  |
| Fig. 13 (tuned) | Random Forest | 95.8 | 95.8 | -0.0 |  |
| Fig. 13 (tuned) | SVC | 89.7 | 91.8 | +2.1 |  |
| Fig. 13 (tuned) | KNN | 92.9 | 90.6 | -2.3 |  |
| Fig. 13 (tuned) | AdaBoost | 92.6 | 93.2 | +0.6 |  |
| Fig. 13 (tuned) | Gradient Boosting | 94.5 | 95.8 | +1.3 |  |
| Fig. 13 (tuned) | CatBoost | 94.9 | 95.8 | +0.9 |  |
| Table 5 (test) | Stack -> Decision Tree | 93.5 | 93.2 | -0.3 |  |
| Table 5 (CV) | Stack -> Decision Tree | 88.0 | 87.3 | -0.7 |  |
| Table 5 (test) | Stack -> Random Forest | 95.5 | 96.2 | +0.7 |  |
| Table 5 (CV) | Stack -> Random Forest | 90.9 | 93.6 | +2.7 |  |
| Table 5 (test) | Stack -> GaussianNB | 94.5 | 92.6 | -1.9 |  |
| Table 5 (CV) | Stack -> GaussianNB | 92.5 | 92.3 | -0.2 |  |
| Table 5 (test) | Stack -> KNN | 94.0 | 95.6 | +1.6 |  |
| Table 5 (CV) | Stack -> KNN | 92.0 | 95.0 | +3.0 |  |
| Table 5 (test) | Stack -> Logistic Regression | 94.5 | 96.2 | +1.7 |  |
| Table 5 (CV) | Stack -> Logistic Regression | 89.9 | 94.8 | +4.9 |  |
| Table 5 (test) | Stack -> SVC | 95.9 | 96.1 | +0.2 |  |
| Table 5 (CV) | Stack -> SVC | 94.8 | 94.6 | -0.2 |  |
| Table 5 (test) | Stack -> AdaBoost | 90.5 | 49.8 | -40.7 |  |
| Table 5 (CV) | Stack -> AdaBoost | 89.0 | 50.7 | -38.3 |  |
| Table 5 (test) | Stack -> Gradient Boosting | 92.5 | 95.5 | +3.0 |  |
| Table 5 (CV) | Stack -> Gradient Boosting | 89.8 | 94.3 | +4.5 |  |
| Table 5 (test) | Stack -> CatBoost | 93.2 | 95.8 | +2.6 |  |
| Table 5 (CV) | Stack -> CatBoost | 93.1 | 93.0 | -0.1 |  |
| Table 6 (deep) | TabNet | 92.0 | 93.6 | +1.6 | 444400 params |
| Table 6 (deep) | CNN | 89.6 | 92.0 | +2.4 | 21206 params; mean over epochs 89.7% |
| Table 6 (deep) | LSTM | 91.8 | 91.2 | -0.6 | 117334 params; mean over epochs 87.4%; hyper-parameters reconstructed |
| Table 6 (deep) | CNN-LSTM | 88.9 | 90.2 | +1.3 | 22582 params; mean over epochs 79.3% |
| Table 6 (deep) | CNN-BiLSTM | 86.8 | 89.7 | +2.9 | 530050 params; mean over epochs 86.4% |

## What reproduces

- **Headline result**: the stacking ensemble with RF + GB + CatBoost base learners and an SVC meta-learner reaches
  **96.1 %** test accuracy and **94.6 %** best cross-validation accuracy (paper: 95.9 % / 94.8 %), with the same best
  parameters (`C=0.1`, linear kernel).
- **Table 4 defaults**: eight of nine classifiers match to the displayed precision, including the deployed
  Random Forest (95.8 %) and Gradient Boosting (95.5 %).
- **Table 6 sizes**: trainable-parameter counts of all five deep models equal the paper exactly (444 400, 21 206,
  117 334, 22 582, 530 050); a test enforces this.
- Dataset: 3 300 x 9, 22 classes x 150, no nulls or duplicates, byte-identical to the published file (a test enforces this).

## What does not, and why

1. **CatBoost default (+2.6 pt).** The notebook's CatBoost cell splits the data *without* stratification. Re-running that
   split with three CatBoost seeds gives 92.6-93.5 %, in line with the paper; the stratified split used here gives
   94.4-94.7 % with the same settings.
2. **Fig. 13 "after tuning" (SVC, KNN, LR, GB, AdaBoost).** In `CSE499RTK_ML2025.ipynb`, cell 41 re-creates `X_train` and
   `X_test` with an extra `StandardScaler` and a non-stratified split, and every later tuning cell silently uses that
   split. Reproducing that state gives SVC 89.7, KNN 92.9, AdaBoost 92.6 and LR 91.2, i.e. the published values.
   The package keeps one consistent protocol instead, so these rows differ.
3. **Table 5, AdaBoost meta-learner (49.8 % vs 90.5 %).** In the notebook (cell 84) this experiment runs on a
   *synthetic two-class* `make_classification` dataset, not on the crop data, so the published figure cannot be obtained on the
   real 22-class data. The same cell also overwrites the global training variables used by the next two cells.
4. **Other Table 5 rows (up to +3 pt on test, +4.9 pt on CV).** The meta-learner cells share variables with the cells above
   them, so their numbers depend on the execution order of the original session, which is not recorded. Base-learner
   settings, search spaces and fold counts here are the ones in the notebook cells (including the per-experiment base
   learners, e.g. 100-tree GB / 500-iteration CatBoost for the Decision-Tree meta-learner, and randomized tuning of the
   base learners for the Random-Forest meta-learner).
5. **Deep models.** Seeds, library versions and CPU vs GPU change training trajectories. For CNN and CNN-BiLSTM the notebooks
   pool predictions over *all* epochs when computing the reported accuracy: the epoch-mean column above (89.7 %, 86.4 %)
   matches Table 6 (89.6 %, 86.8 %), while the final-epoch accuracy reported in the main column is higher. TabNet in the
   notebook used its library-default seed (0) on a GPU and recorded 91.97 %; here it reaches about 93.5 %. The LSTM
   notebook is not in the repository, so its learning rate and epoch count are reconstructed from the paper (2 layers x 96
   units, 117 334 parameters).
6. **LIME.** Section 3.4 (PCA + Random Forest), Fig. 24 (Gradient Boosting) and the notebook (best of RF/GB/LR on 7
   features) disagree with each other; `crop-rec explain` works with any saved model that has `predict_proba`.

Because items 1-5 stem from the original notebooks' execution state rather than from the method described in the paper,
the package deliberately keeps a single, stated protocol rather than reproducing the notebook state.

