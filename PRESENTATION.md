# Project 1 — Presentation Notes

## Problem
Estimate cutting-force components during longitudinal turning using acoustic features and machining settings.

## Dataset
- 100 machining trials
- 260 extracted sound features per trial
- 3 machining parameters: cutting speed, cutting depth, feed rate
- 3 measured outputs: Ft, Fr and Fa

Source: [Mendeley Data](https://data.mendeley.com/datasets/y89s8vn8r4/1)

## Approach
- Data validation and feature assembly
- StandardScaler + PCA + multi-output MLP
- Ridge Regression and Random Forest benchmarks
- 80/20 holdout test and 5-fold cross-validation
- MAE, RMSE and R² evaluation

## Main result
Random Forest achieved the strongest result:
- Holdout MAE: 90.78
- Holdout RMSE: 140.88
- Holdout R²: 0.681
- 5-fold mean MAE: 76.96
- 5-fold mean R²: 0.680

The MLP neural network was also trained and evaluated, but did not outperform Random Forest. This is reported honestly as part of the experiment.

## Key limitation
There are only 100 trials. The cutting-parameter file does not contain a trial ID and is aligned by the dataset's supplied row order. This mapping should be verified from the source documentation. The results are a portfolio/research prototype, not proof of production readiness.
