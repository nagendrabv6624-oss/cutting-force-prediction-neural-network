"""
Train and validate a multi-output neural network for cutting-force prediction.

Dataset: 100 machining trials:
  sound_features.csv       260 extracted acoustic features + sound filename
  force_features.csv       measured force components Ft, Fr, Fa + force filename
  cutting_parameters.csv   cutting_speed, cutting_depth, feed_rate

The parameter file has no explicit trial ID. The script verifies sequential
sound/force identifiers and retains the parameter file's supplied row order.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["Ft", "Fr", "Fa"]
PARAMS = ["cutting_speed", "cutting_depth", "feed_rate"]


def trial_number(value: object) -> int:
    match = re.search(r"(\\d+)", str(value))
    if not match:
        raise ValueError(f"Could not extract trial number from identifier: {value}")
    return int(match.group(1))


def load_dataset(sound_path: Path, force_path: Path, params_path: Path):
    sound = pd.read_csv(sound_path)
    force = pd.read_csv(force_path)
    params = pd.read_csv(params_path)

    for name, frame, required in [
        ("sound", sound, ["Unnamed: 0"]),
        ("force", force, ["Unnamed: 0"] + TARGETS),
        ("parameters", params, PARAMS),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} file is missing columns: {sorted(missing)}")

    sound_ids = sound["Unnamed: 0"].map(trial_number).to_numpy()
    force_ids = force["Unnamed: 0"].map(trial_number).to_numpy()
    expected = np.arange(1, len(sound) + 1)
    if len(sound) != 100 or len(force) != 100 or len(params) != 100:
        raise ValueError("Expected exactly 100 rows in each supplied CSV.")
    if not np.array_equal(sound_ids, expected):
        raise ValueError("Sound IDs must be sequential from 001 to 100.")
    if not np.array_equal(force_ids, expected):
        raise ValueError("Force IDs must be sequential from 001 to 100.")
    if len(set(sound_ids)) != len(sound_ids) or len(set(force_ids)) != len(force_ids):
        raise ValueError("Duplicate trial IDs detected.")

    sound_features = sound.drop(columns=["Unnamed: 0"]).select_dtypes(include=[np.number])
    if sound_features.shape[1] != 260:
        raise ValueError(f"Expected 260 numeric sound features; found {sound_features.shape[1]}.")
    if (sound_features.isna().any().any() or force[TARGETS].isna().any().any()
            or params[PARAMS].isna().any().any()):
        raise ValueError("Missing values found. Resolve them before training.")
    if not np.isfinite(sound_features.to_numpy()).all():
        raise ValueError("Non-finite values found in sound features.")

    # Parameter CSV has no trial ID; preserve its documented dataset order.
    X = pd.concat(
        [sound_features.reset_index(drop=True), params[PARAMS].reset_index(drop=True)],
        axis=1,
    )
    y = force[TARGETS].reset_index(drop=True).astype(float)
    return X, y, sound_features.columns.tolist()


def build_models():
    # Regularized MLP + PCA limits model complexity for 100 rows / 263 predictors.
    neural_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=42)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
            alpha=1.0, learning_rate_init=0.001, max_iter=1000,
            early_stopping=False, random_state=42, tol=1e-5,
            n_iter_no_change=50,
        )),
    ])
    neural_net = TransformedTargetRegressor(
        regressor=neural_pipeline, transformer=StandardScaler()
    )
    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=42)),
        ("ridge", Ridge(alpha=10.0)),
    ])
    forest = RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, max_features=0.7,
        random_state=42, n_jobs=-1,
    )
    return {
        "neural_network": neural_net,
        "ridge_baseline": ridge,
        "random_forest": forest,
    }


def metric_pack(y_true, y_pred):
    result = {
        "MAE_average": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_average": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2_uniform_average": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "per_target": {},
    }
    for i, target in enumerate(TARGETS):
        result["per_target"][target] = {
            "MAE": float(mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])),
            "RMSE": float(np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))),
            "R2": float(r2_score(y_true.iloc[:, i], y_pred[:, i])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Train cutting-force regression models.")
    parser.add_argument("--sound", default="data/sound_features.csv")
    parser.add_argument("--force", default="data/force_features.csv")
    parser.add_argument("--parameters", default="data/cutting_parameters.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    X, y, sound_columns = load_dataset(
        Path(args.sound), Path(args.force), Path(args.parameters)
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    models = build_models()
    fitted, predictions, test_metrics = {}, {}, {}

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        fitted[name] = estimator
        predictions[name] = pred
        test_metrics[name] = metric_pack(y_test, pred)

    # Preprocessing remains inside each estimator pipeline to avoid CV leakage.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    for name, estimator in models.items():
        scores = cross_validate(
            clone(estimator), X, y, cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1, error_score="raise",
        )
        cv_summary[name] = {
            "5_fold_MAE_mean": float(-scores["test_mae"].mean()),
            "5_fold_MAE_std": float(scores["test_mae"].std()),
            "5_fold_R2_mean": float(scores["test_r2"].mean()),
            "5_fold_R2_std": float(scores["test_r2"].std()),
        }

    out, models_dir = ROOT / "outputs", ROOT / "models"
    out.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    best_name = min(cv_summary, key=lambda key: cv_summary[key]["5_fold_MAE_mean"])
    joblib.dump(fitted["neural_network"], models_dir / "cutting_force_mlp.joblib")
    joblib.dump(fitted[best_name], models_dir / "best_validated_model.joblib")

    metrics = {
        "dataset": {
            "trials": int(len(X)), "sound_features": int(len(sound_columns)),
            "cutting_parameters": int(len(PARAMS)), "targets": TARGETS,
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "split": "80/20 random holdout, random_state=42",
            "note": "Cutting parameters have no trial ID and are aligned by supplied row order.",
        },
        "holdout_test": test_metrics,
        "five_fold_cross_validation": cv_summary,
        "best_model_by_cv_mae": best_name,
    }
    (out / "feature_columns.json").write_text(json.dumps(X.columns.tolist(), indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    holdout = y_test.reset_index(drop=True).copy()
    for i, target in enumerate(TARGETS):
        holdout[f"{target}_predicted_mlp"] = predictions["neural_network"][:, i]
        holdout[f"{target}_residual_mlp"] = (
            holdout[target] - holdout[f"{target}_predicted_mlp"]
        )
    holdout.to_csv(out / "holdout_predictions.csv", index=False)

    plt.figure(figsize=(9, 5))
    y.melt(var_name="Force component", value_name="Measured force").boxplot(
        column="Measured force", by="Force component", grid=False
    )
    plt.suptitle("")
    plt.title("Measured cutting-force distributions")
    plt.tight_layout()
    plt.savefig(out / "force_distributions.png", dpi=160)
    plt.close()

    mlp = fitted["neural_network"].regressor_.named_steps["mlp"]
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("Neural-network training loss")
    plt.tight_layout()
    plt.savefig(out / "training_loss.png", dpi=160)
    plt.close()

    for i, target in enumerate(TARGETS):
        actual = y_test.iloc[:, i].to_numpy()
        predicted = predictions["neural_network"][:, i]
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        plt.figure(figsize=(6, 6))
        plt.scatter(actual, predicted, alpha=0.8)
        plt.plot([low, high], [low, high], linestyle="--")
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"Actual vs predicted: {target}")
        plt.tight_layout()
        plt.savefig(out / f"actual_vs_predicted_{target}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.hist(actual - predicted, bins=10)
        plt.xlabel(f"{target} residual (actual - predicted)")
        plt.ylabel("Count")
        plt.title(f"Residual distribution: {target}")
        plt.tight_layout()
        plt.savefig(out / f"residuals_{target}.png", dpi=160)
        plt.close()

    names = list(cv_summary)
    values = [cv_summary[n]["5_fold_MAE_mean"] for n in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("Mean 5-fold MAE (lower is better)")
    plt.title("Model comparison using 5-fold cross-validation")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "model_comparison_cv.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved models to: {models_dir}")
    print(f"Saved reports and graphs to: {out}")


if __name__ == "__main__":
    main()
