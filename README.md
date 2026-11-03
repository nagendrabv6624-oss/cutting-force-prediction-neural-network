# Cutting Force Prediction from Acoustic Features

A multi-output machine-learning project that estimates three measured cutting-force components from processed sound features and machining parameters.

**Dataset:** [Mendeley Data — Sound and Cutting Force Components in Longitudinal Turning of C45 Steel](https://data.mendeley.com/datasets/y89s8vn8r4/1)

## Project objective

Predict the three force components:
- **Ft** — tangential cutting force
- **Fr** — radial cutting force
- **Fa** — axial cutting force

The feature matrix contains 260 extracted acoustic features and 3 machining parameters (cutting speed, cutting depth and feed rate).

## Methodology

1. Validate the three source CSVs and trial identifiers.
2. Standardize input features and reduce dimensionality with PCA inside each model pipeline.
3. Train a multi-output MLP neural network.
4. Compare it against Ridge Regression and Random Forest baselines.
5. Evaluate with an 80/20 holdout split and 5-fold cross-validation.
6. Export model artifacts, metrics, predictions and diagnostic plots.

## Dataset

Download the three processed CSV files from the dataset's `datasets` folder:
- `sound_features.csv`
- `force_features.csv`
- `cutting_parameters.csv`

Place them in `data/`. The large raw-audio archive is not needed for this workflow. See [data/README.md](data/README.md) for schema and alignment notes. The source data are not mirrored in this repository; follow the dataset's current licence and attribution terms.

**Alignment caveat:** The parameter CSV has no explicit trial identifier. Its rows are paired by the supplied order with the sound/force records. The script verifies sequential sound and force identifiers, but the parameter-row mapping should be checked against the source documentation before making scientific or production claims.

## Results from the supplied 100-trial experiment

The following metrics were produced by executing the training pipeline on the three supplied CSV files (80 training rows, 20 holdout rows, seed 42).

### Holdout test set

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| MLP neural network | 156.77 | 239.35 | 0.336 |
| Ridge Regression | 160.08 | 248.89 | 0.373 |
| **Random Forest** | **90.78** | **140.88** | **0.681** |

### Five-fold cross-validation

| Model | Mean MAE ↓ | MAE std | Mean R² ↑ |
|---|---:|---:|---:|
| MLP neural network | 173.83 | 21.61 | 0.084 |
| Ridge Regression | 155.79 | 14.59 | 0.307 |
| **Random Forest** | **76.96** | **10.47** | **0.680** |

**Finding:** Random Forest was the best-performing model in this experiment. The MLP was trained, but did not outperform the Random Forest. The comparison is reported transparently rather than presenting the neural network as the winner.

Per-target metrics and run details are in [outputs/metrics.json](outputs/metrics.json). Holdout predictions from the MLP are in [outputs/holdout_predictions.csv](outputs/holdout_predictions.csv).

## Repository structure

```text
data/       Dataset instructions and expected CSV files
src/        Training and inference scripts
outputs/    Evaluation metrics, feature schema and holdout predictions
models/     Generated model artifacts (created by the training script)
```

## Run the project

Python 3.10–3.12 is recommended.

```bash
pip install -r requirements.txt
python src/train.py
```

Optional custom data paths:

```bash
python src/train.py --sound data/sound_features.csv --force data/force_features.csv --parameters data/cutting_parameters.csv
```

To predict new records after training, provide a CSV containing the same 263 input columns:

```bash
python src/predict.py --input path/to/new_feature_rows.csv
```

The inference script writes `outputs/predictions_for_new_data.csv`.

## Limitations and responsible interpretation

- The experiment contains only 100 trials; generalization estimates are uncertain.
- The holdout set contains 20 rows. Five-fold CV is an internal stability check, not external validation.
- Parameter-row alignment depends on the source dataset's ordering and needs confirmation from its documentation.
- The MLP reached its configured iteration limit in some fits and has a weaker cross-validation score than Random Forest.
- Force units should be confirmed from the dataset documentation before interpreting absolute errors.
- This is a research/portfolio prototype, not a calibrated industrial control or safety system.

