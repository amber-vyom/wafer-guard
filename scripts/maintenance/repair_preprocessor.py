from pathlib import Path
import joblib
import pandas as pd

root = Path(__file__).resolve().parents[2]
artifact_path = root / "artifacts" / "preprocessor.joblib"
data_path = root / "data" / "raw" / "secom_telemetry_modified.csv"

preprocessor = joblib.load(artifact_path)
df = pd.read_csv(data_path)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
usable_cols = preprocessor["usable_cols"]
flag_cols = preprocessor["flag_cols"]

feature_cols = list(usable_cols) + list(flag_cols)

preprocessor["sensor_cols"] = sensor_cols
preprocessor["feature_cols"] = feature_cols

backup_path = root / "artifacts" / "preprocessor_before_metadata_fix.joblib"
joblib.dump(joblib.load(artifact_path), backup_path)
joblib.dump(preprocessor, artifact_path)

print("Preprocessor metadata repaired.")
print("Sensor columns:", len(sensor_cols))
print("Usable columns:", len(usable_cols))
print("Flag columns:", len(flag_cols))
print("Expected features:", len(feature_cols))
print("Backup:", backup_path)
