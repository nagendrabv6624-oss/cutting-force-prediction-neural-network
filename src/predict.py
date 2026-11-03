"""Run inference with the cross-validation-selected cutting-force model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="CSV containing the 260 sound features and 3 cutting parameters",
    )
    parser.add_argument("--output", default="outputs/predictions_for_new_data.csv")
    args = parser.parse_args()

    model_path = ROOT / "models" / "best_validated_model.joblib"
    feature_path = ROOT / "outputs" / "feature_columns.json"
    if not model_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Train the model first with: python src/train.py")

    features = json.loads(feature_path.read_text())
    frame = pd.read_csv(args.input)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required columns. "
            f"First missing columns: {missing[:10]}"
        )

    X = frame[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Input contains missing or non-numeric feature values.")

    model = joblib.load(model_path)
    prediction = model.predict(X)
    result = pd.DataFrame(
        prediction, columns=["Ft_predicted", "Fr_predicted", "Fa_predicted"]
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.head())
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
