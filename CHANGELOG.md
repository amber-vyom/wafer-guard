# Changelog

## Unreleased (cleanup, October 2026)

### Changed
- One chronological split (70% train, 15% validation, 15% test) in `src/data/splits.py`, used by the notebook, the app and the evaluation scripts. The app previously scored the last 20%, which overlapped the validation rows. **Reported test figures changed**: 236 rows with 9 defects, instead of 314 with 17.
- Stable timestamp sort, so rows with repeated timestamps keep a fixed order.
- One model class, `WaferAutoencoder`, shared by the notebook and the app; its width is configurable and old checkpoints still load.
- The notebook saves artifacts in the format the app loads (with timestamped backups) and ends with a parity check. Its release gate now uses lift plus the average-precision confidence bound instead of F1 >= 0.90.
- The loader validates artifact keys and reports clear errors; the app caches data and the model and shows alert rate and prevalence.
- `AUDIT.md` metrics replaced with saved notebook outputs, plus findings on signal and drift.

### Added
- Pytest suite in `tests/` (synthetic data), `docs/`, `CONTRIBUTING.md`, `.editorconfig`, `pytest.ini`.
- Evaluation report JSON records the split and model artifact.

### Removed / moved
- Duplicate scripts in the repository root.
- One-off experiments from `src/models/` and `src/evaluation/` moved to `archive/experiments/`.
- Unused supervised-candidate helpers in `prediction_service.py`.

### Fixed
- Scripts under `scripts/` resolved the wrong project root after being moved.
- The saved `preprocessor.joblib` lacked `sensor_cols` and `feature_cols`; added with `scripts/maintenance/repair_preprocessor.py` (original kept as `preprocessor_before_metadata_fix.joblib`).
