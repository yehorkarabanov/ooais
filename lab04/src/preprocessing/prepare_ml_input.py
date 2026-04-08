import json
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

PROCESSED_DATA_PATH = DATA_PATH / "processed"
VALID_DATA_PATH = PROCESSED_DATA_PATH / "observations_valid.csv"
INVALID_DATA_PATH = PROCESSED_DATA_PATH / "observations_invalid.csv"
MODEL_INPUT_PATH = PROCESSED_DATA_PATH / "model_input.csv"

MODEL_FEATURES_PATH = PROCESSED_DATA_PATH / "model_features.csv"
MODEL_LABELS_PATH = PROCESSED_DATA_PATH / "model_labels.csv"

REPORTS_PATH = PROJECT_ROOT / "reports"

with open(VALID_DATA_PATH) as data_file:
    data_raw = csv.DictReader(data_file)
    data_raw = list(data_raw)

REQUIRED_NUMERIC_COLUMNS = [
    "temperature",
    "velocity",
    "altitude",
    "signal_strength",
]
rec_acc = []
rec_rej = []

for row in data_raw:
    tmp = row
    for key in REQUIRED_NUMERIC_COLUMNS:
        if key not in row:
            rec_rej.append(tmp)
            continue
        try:
            tmp[key] = float(row[key])
        except ValueError:
            rec_rej.append(tmp)
            continue
    if tmp["altitude"] < 0:
        rec_rej.append(tmp)
        continue
    rec_acc.append(tmp)

print("=== ML Input Preparation: Loading and Conversion ===")
print(f"Input file: {VALID_DATA_PATH.relative_to(PROJECT_ROOT)}")
print(f"Records loaded: {len(data_raw)}")
print(f"Records accepted: {len(rec_acc)}")
print(f"Records rejected: {len(rec_rej)}")

data = rec_acc

min_max_val = []
for key in REQUIRED_NUMERIC_COLUMNS:
    minv = min(d[key] for d in rec_acc)
    maxv = max(d[key] for d in rec_acc)
    min_max_val.append([minv, maxv])

for row in data:
    for key in REQUIRED_NUMERIC_COLUMNS:
        row[key] = (row[key] - min_max_val[REQUIRED_NUMERIC_COLUMNS.index(key)][0]) / (
                    min_max_val[REQUIRED_NUMERIC_COLUMNS.index(key)][1] -
                    min_max_val[REQUIRED_NUMERIC_COLUMNS.index(key)][0])
print("\n=== ML Input Preparation: Normalization ===")
print("Normalization completed successfully.")
print("All selected numerical features are in range [0,1].")


for row in data:
    row["temperature_velocity_interaction"] = row["temperature"] * row["velocity"]
    row["altitude_signal_ration"] = row["altitude"] / (row["signal_strength"] + 1e-4)

print("=== ML Input Preparation: Derived Features ===")
print("New features added:\n- temperature_velocity_interaction\n- altitude_signal_ratio")
print("Example record (extended):")
print(json.dumps(data[0], indent=2))

for row in data:
    dt_obj = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
    row["hour_normalized"] = dt_obj.hour / 24

print("""=== ML Input Preparation: Temporal Features ===
New feature added:
- hour_normalized
Example record (extended):""")
print(json.dumps(data[0], indent=2))

old_data = data
data = []
for row in old_data:
    data.append({
        "temperature": row["temperature"],
        "velocity": row["velocity"],
        "altitude": row["altitude"],
        "signal_strength": row["signal_strength"],
        "temperature_velocity_interaction": row["temperature_velocity_interaction"],
        "altitude_signal_ration": row["altitude_signal_ration"],
        "hour_normalized": row["hour_normalized"],
    })

print("""
=== ML Input Preparation: Feature Selection ===
Selected features:
- temperature
- velocity
- altitude
- signal_strength
- temperature_velocity_interaction
- altitude_signal_ratio
- hour_normalized

Example record (final):
""")
print(json.dumps(data[0], indent=2))

model_features = data
model_labels = [{"anomaly_flag": d["anomaly_flag"]} for d in old_data]

with open(MODEL_FEATURES_PATH, "w", newline="") as features_file:
    writer = csv.DictWriter(features_file, fieldnames=model_features[0].keys())
    writer.writeheader()
    writer.writerows(model_features)

with open(MODEL_LABELS_PATH, "w", newline="") as labels_file:
    writer = csv.DictWriter(labels_file, fieldnames=model_labels[0].keys())
    writer.writeheader()
    writer.writerows(model_labels)

print(f"""=== ML Input Preparation: Saving Outputs ===
Saved file: {MODEL_FEATURES_PATH.relative_to(PROJECT_ROOT)}
Saved file: {MODEL_LABELS_PATH.relative_to(PROJECT_ROOT)}

Number of records: {len(model_features)}
Number of features: {len(model_features[0])}

Example label record:
{model_labels[0]}
""")

