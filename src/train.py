"""Train an MLP regressor for machining cutting-force estimation.

Without --data, a synthetic demonstration dataset is generated. Its metrics
must not be interpreted as evidence of real-world machining performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["cutting_speed", "feed_rate", "depth_of_cut", "tool_wear"]
TARGET = "cutting_force"


def make_demo_data(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create illustrative synthetic data; not a physical machining model."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(50, 250, n)
    feed = rng.uniform(0.05, 0.35, n)
    depth = rng.uniform(0.2, 3.0, n)
    wear = rng.uniform(0, 120, n)
    force = (
        120 + 1.8 * speed + 850 * feed + 95 * depth
        + 0.35 * wear + 180 * feed * depth
        + rng.normal(0, 35, n)
    )
    return pd.DataFrame({
        "cutting_speed": speed, "feed_rate": feed,
        "depth_of_cut": depth, "tool_wear": wear,
        "cutting_force": force,
    })


def load_data(path: str | None) -> tuple[pd.DataFrame, bool]:
    if path:
        frame = pd.read_csv(path)
        synthetic = False
    else:
        frame = make_demo_data()
        synthetic = True
        print("Using SYNTHETIC demonstration data; results are not industrial evidence.")
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        raise ValueError("At least 30 complete rows are required.")
    return frame, synthetic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="CSV path; omit for synthetic demo data")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)
    frame, synthetic = load_data(args.data)
    X, y = frame[FEATURES].to_numpy(), frame[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        )
    ]
    model.fit(X_train, y_train, validation_split=0.2, epochs=args.epochs,
              batch_size=32, callbacks=callbacks, verbose=1)

    predictions = model.predict(X_test, verbose=0).ravel()
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "data_type": "synthetic_demo" if synthetic else "user_provided",
        "test_rows": int(len(y_test)),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, predictions)),
    }

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    model.save(out / "cutting_force_model.keras")
    joblib.dump(scaler, out / "scaler.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.6)
    low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual cutting force")
    plt.ylabel("Predicted cutting force")
    plt.title("Actual vs predicted cutting force")
    plt.tight_layout()
    plt.savefig(out / "predictions.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
