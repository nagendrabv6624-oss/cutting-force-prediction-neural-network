# Cutting Force Prediction Using a Neural Network

A portfolio project demonstrating regression with a TensorFlow/Keras multilayer perceptron (MLP) to estimate machining cutting force from process parameters.

> **Data transparency:** This repository does not contain proprietary or measured industrial data. If no CSV is supplied, the training script creates a clearly labelled synthetic demonstration dataset. Do not present synthetic-data metrics as real machining results.

## Project workflow
1. Load a user-provided machining dataset, or generate synthetic demonstration data.
2. Validate columns and prepare features.
3. Split data into training and test sets.
4. Standardize input features.
5. Train an MLP regressor with early stopping.
6. Evaluate using MAE, RMSE and R²; save the model and a prediction plot.

## Expected CSV format
Provide `data/cutting_force.csv` with these numeric columns:

- `cutting_speed` (m/min)
- `feed_rate` (mm/rev)
- `depth_of_cut` (mm)
- `tool_wear` (minutes or a consistent wear measure)
- `cutting_force` (target, N)

The feature units and meaning must be consistent throughout your dataset.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

To use another CSV:
```bash
python src/train.py --data path/to/your_data.csv
```

## Outputs
The script creates `outputs/cutting_force_model.keras`, `outputs/scaler.joblib`, `outputs/metrics.json` and `outputs/predictions.png`.

## Important limitations
- Synthetic data is only for demonstrating the end-to-end workflow.
- A model trained on synthetic data is not validated for real machine settings.
- For credible industrial conclusions, use a documented real dataset, explain its source and licensing, check units, and evaluate on a representative holdout set.
- Tune architecture and hyperparameters only using training/validation data; retain a final untouched test set for final reporting.

## Skills demonstrated
Python, Pandas, NumPy, data validation, regression, TensorFlow/Keras, model evaluation and Matplotlib.
