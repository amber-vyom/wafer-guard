# Artifact format

The notebook writes these files and `src/services/prediction_service.py` loads them. They are
gitignored; a fresh checkout must generate them (see the README) or have them supplied.

## `artifacts/wafer_model.pt`

Loaded with `torch.load(..., weights_only=True)`, so it holds only tensors and plain Python values.

| Key | Type | Meaning |
|---|---|---|
| `model_state_dict` | dict of tensors | Weights for `WaferAutoencoder` |
| `input_dim` | int | Number of input features (continuous sensors + indicator columns) |
| `latent_dim` | int | Latent width (default 32 if missing) |
| `hidden_dim` | int | First hidden width (default 128 if missing, for older checkpoints) |
| `threshold` | float | Alert threshold on the sigmoid score, chosen on validation data |

Required: `model_state_dict`, `input_dim`, `threshold`. The notebook also stores `usable_cols`,
`flag_cols`, `target_alert_rate`, `best_epoch` and `best_validation_ap` for provenance.

## `artifacts/preprocessor.joblib`

A dict loaded with `joblib`.

| Key | Meaning |
|---|---|
| `imputer` | `SimpleImputer(strategy="median")` fitted on training rows only |
| `scaler` | `StandardScaler` fitted on the imputed training rows |
| `sensor_cols` | Every raw `sensor_*` column the input must contain |
| `usable_cols` | Columns passed to the imputer and scaler, in order |
| `flag_cols` | Indicator columns appended after the scaled features |
| `feature_cols` | `usable_cols + flag_cols`; its length must equal `input_dim` |

All six keys are required. The loader raises a clear error if one is missing.

## Feature layout

`[scaled usable_cols..., is_legacy_ps_l, legacy_sensor_1_code_100, legacy_sensor_1_code_200, legacy_sensor_1_code_300]`

- `sensor_0 == 0.0` marks a legacy wafer. On those rows `sensor_0` and `sensor_1` are treated as missing.
- The three code flags are 1 only on legacy rows whose `sensor_1` equals 100, 200 or 300.

## Two coexisting versions

| | Deployed (older) | Notebook output (newer) |
|---|---|---|
| `usable_cols` | 468 sensors, including `sensor_0` and `sensor_1` | 588 sensors, excluding `sensor_0` and `sensor_1` |
| Features | 472 | 592 |
| Threshold | 0.001003 | chosen for a 10% alert rate on each run |

Both load through the same code. Re-running the notebook replaces the deployed files after saving
timestamped `.bak` copies; set `WAFERGUARD_ARTIFACT_DIR` to write elsewhere.

## Gradient-boosting candidate

`artifacts/experiments/wafer_guard_final_gb.joblib` is a separate dict written by
`scripts/training/train_model.py` (keys include `model`, `imputer`, `usable_cols`, `flag_cols`,
`threshold`). Its threshold of 0.5 is a provisional placeholder.
