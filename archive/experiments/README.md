# Archived experiments

One-off diagnostics and model experiments that are not part of the application.
They were moved here from `src/models/` and `src/evaluation/`; their results live in
`reports/experiments/` and `artifacts/experiments/`.

They are kept for reference and are not maintained or tested. Run them from the
repository root with the root on the import path, for example:

```powershell
$env:PYTHONPATH = "."
python archive/experiments/<script>.py
```

Some scripts import each other as `src.models.<name>` (the old location) or hard-code
older 70/10/20 split boundaries. The supported split is defined in `src/data/splits.py`.
