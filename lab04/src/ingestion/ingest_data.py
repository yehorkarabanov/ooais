import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
DATASET_PATH = RAW_DATA_PATH / "orbital_observations.csv"
METADATA_PATH = RAW_DATA_PATH / "metadata.json"

PROCESSED_DATA_PATH = DATA_PATH / "processed"
VALID_DATA_PATH = PROCESSED_DATA_PATH / "observations_valid.csv"
INVALID_DATA_PATH = PROCESSED_DATA_PATH / "observations_invalid.csv"
MODEL_INPUT_PATH = PROCESSED_DATA_PATH / "model_input.csv"

REPORTS_PATH = PROJECT_ROOT / "reports"
INGESTION_SUMMARY_PATH = REPORTS_PATH / "ingestion_summary.txt"

with open(METADATA_PATH) as metadata_file:
    metadata = json.load(metadata_file)

with open(DATASET_PATH) as data_file:
    rows = csv.DictReader(data_file)
    rows = list(rows)

print("\nTask 4")
print(f"Dataset name: {metadata.get('dataset_name', 'unknown')}")
print(f"Number of rows: {len(rows)}")
print(f"Column names in dataset  {list(rows[0].keys())}")
print(f"Column names in metadata {metadata.get('columns', {})}")

print("\nTask 5")
column_validation_ok = set(rows[0].keys()) == set(metadata.get("columns", {}))
if not column_validation_ok:
    print("Column validation: MISMATCH")
    print(f"Expected: {set(rows[0].keys())}")
    print(f"Actual: {set(metadata.get('columns', {}))}")
else:
    print("Column validation: OK")

print("\nTask 6")
loaded_records = len(rows)
expected_records = metadata.get("num_records")
record_count_validation_ok = loaded_records == expected_records
print(f"Number of rows in dataset: {loaded_records}")
print(f"Number of rows in metadata: {expected_records}")
if not record_count_validation_ok:
    print("Row count validation: MISMATCH")
    print(f"Expected: {expected_records}")
    print(f"Actual: {loaded_records}")
else:
    print("Row count validation: OK")

print("\nTask 7")
valid = []
invalid = []
for row in rows:
    if row.get("temperature") == "INVALID" or row.get("sensor_status") != "OK":
        invalid.append(row)
        continue

    valid.append(row)

print(f"Valid rows: {len(valid)}")
print(f"Invalid rows: {len(invalid)}")

print("\nTask 8")
print("Saving valid and invalid data to separate CSV files...")
with open(VALID_DATA_PATH, "w", newline="") as valid_file:
    writer = csv.DictWriter(valid_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(valid)

with open(INVALID_DATA_PATH, "w", newline="") as invalid_file:
    writer = csv.DictWriter(invalid_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(invalid)

print("Data saved successfully.")

print("\nTask 9")
model_input = [
    {k: row.get(k) for k in metadata.get("feature_columns")} for row in valid
]
print("Saving model input data to CSV file...")
with open(MODEL_INPUT_PATH, "w", newline="") as model_input_file:
    writer = csv.DictWriter(
        model_input_file, fieldnames=metadata.get("feature_columns")
    )
    writer.writeheader()
    writer.writerows(model_input)

print("Data saved successfully.")

print("\nTask 10")
generated_files = [
    str(VALID_DATA_PATH),
    str(INVALID_DATA_PATH),
    str(MODEL_INPUT_PATH),
    str(INGESTION_SUMMARY_PATH),
]

with open(INGESTION_SUMMARY_PATH, "w") as summary_file:
    summary_file.write(f"Dataset name: {metadata.get('dataset_name', 'unknown')}\n")
    summary_file.write(f"Number of loaded records: {loaded_records}\n")
    summary_file.write(f"Expected number of records: {expected_records}\n")
    summary_file.write(
        "Column validation result: OK\n"
        if column_validation_ok
        else "Column validation result: MISMATCH\n"
    )
    summary_file.write(
        "Record count validation result: OK\n"
        if record_count_validation_ok
        else "Record count validation result: MISMATCH\n"
    )
    summary_file.write(f"Number of valid records: {len(valid)}\n")
    summary_file.write(f"Number of invalid records: {len(invalid)}\n")
    summary_file.write("Generated output files:\n")
    for output_file in generated_files:
        summary_file.write(f"- {output_file}\n")
