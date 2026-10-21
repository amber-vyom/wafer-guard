# Contributing

- Install pinned dependencies: `pip install -r requirements.txt`.
- Run `python -m pytest` before committing. The tests use synthetic data and need neither the dataset nor the artifacts. Also run `python src/evaluation/smoke_test.py` if you have the artifacts.
- Never commit data or model artifacts (`data/raw/*.csv`, `artifacts/`, `*.pt`, `*.joblib`); they are gitignored.
- Change the train/validation/test boundaries only in `src/data/splits.py`, and update the README, `AUDIT.md` and `CHANGELOG.md` figures when you do.
- Keep the notebook and the app compatible: if you change the artifact keys or feature layout, update `docs/artifacts.md` and the parity check.
- Report metrics from a saved run, with the checkpoint, threshold and alert rate. Treat test numbers as exploratory.
- Use UTF-8 and the settings in `.editorconfig`.
