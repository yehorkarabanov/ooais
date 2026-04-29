import csv
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

PROCESSED_DATA_PATH = DATA_PATH / "processed"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

MODEL_FEATURES_PATH = PROCESSED_DATA_PATH / "model_features.csv"
MODEL_LABELS_PATH = PROCESSED_DATA_PATH / "model_labels.csv"

RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_PATH / "decision_tree_model.joblib"
MODEL_EVALUATION_PATH = RESULTS_PATH / "model_evaluation.txt"

REPORTS_PATH: Path = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_TRAINING_SUMMARY_PATH = REPORTS_PATH / "model_training_summary.txt"

with open(MODEL_FEATURES_PATH) as data_file:
    features = csv.DictReader(data_file)
    features = list(features)

with open(MODEL_LABELS_PATH) as data_file:
    labels = csv.DictReader(data_file)
    labels = list(labels)

print(f"""
=== Machine Learning: Loading Feature Dataset ===
Input file: {MODEL_FEATURES_PATH.relative_to(PROJECT_ROOT)}
Records loaded: {len(features)}
Columns: {features[0].keys() if features else "N/A"}
""")

x = []
for feature in features:
    x.append(list(float(t) for t in list(feature.values())))

y = []
for label in labels:
    y.append(int(label["anomaly_flag"]))

print(f"""
=== Machine Learning: Preparing Features and Target ===
Number of samples in X: {len(x)}
Number of labels in y: {len(y)}
Target values detected: [0, 1]
""")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"""
=== Machine Learning: Train/Test Split ===
Training samples: {len(x_train)}
Test samples: {len(x_test)}
""")

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

print(f"""
=== Machine Learning: Model Training ===
Model: {model.__class__.__name__}
Training completed successfully.
""")

predictions = model.predict(x_test)
print(f"""
=== Machine Learning: Prediction ===
Predictions generated for test set.
Number of predictions: {len(predictions)}
Example predictions:
{predictions[:10]}
""")

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(f"""
=== Machine Learning: Evaluation ===
Accuracy: f{accuracy}
Confusion Matrix:
{conf_matrix}
""")

with open(MODEL_PATH, "wb") as model_file:
    joblib.dump(model, model_file)

tree_rules = export_text(
    model, feature_names=[f"feature_{i}" for i in range(len(x[0]))]
)
print(f"""
=== Machine Learning: Saving and Inspecting Model ===
Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}
Model type: {model.__class__.__name__}
Tree depth: {model.get_depth()}
Number of leaves: {model.get_n_leaves()}
Decision Tree Rules:
{tree_rules}
""")

with open(MODEL_EVALUATION_PATH, "w") as eval_file:
    eval_file.write("OOAIS Model Evaluation\n")
    eval_file.write("====================== \n\n")
    eval_file.write(f"Model: {model.__class__.__name__}\n")
    eval_file.write(f"Training samples: {len(x_train)}\n")
    eval_file.write(f"Test samples: {len(x_test)}\n\n")
    eval_file.write(f"Accuracy: {accuracy}\n")
    eval_file.write("Confusion Matrix:\n")
    eval_file.write(f"{conf_matrix}\n\n")
    eval_file.write("Decision Tree Rules:\n")
    eval_file.write(f"{tree_rules}\n")

with open(MODEL_TRAINING_SUMMARY_PATH, "w") as summary_file:
    summary_file.write("OOAIS Model Training Summary\n")
    summary_file.write("====================== \n\n")
    summary_file.write("Input datasets\n")
    summary_file.write("--------------\n")
    summary_file.write(f"{MODEL_FEATURES_PATH.relative_to(PROJECT_ROOT)}\n")
    summary_file.write(f"{MODEL_LABELS_PATH.relative_to(PROJECT_ROOT)}\n\n")
    summary_file.write("Dataset statistics\n")
    summary_file.write("------------------\n")
    summary_file.write(f"Number of samples: {len(x)}\n")
    summary_file.write(f"Number of features: {len(x[0])}\n\n")
    summary_file.write("Model\n")
    summary_file.write("-----\n")
    summary_file.write(f"{model.__class__.__name__}\n\n")
    summary_file.write("Train/Test split\n")
    summary_file.write("----------------\n")
    summary_file.write(f"Training samples: {len(x_train)}\n")
    summary_file.write(f"Test samples: {len(x_test)}\n\n")
    summary_file.write("Evaluation summary\n")
    summary_file.write("------------------\n")
    summary_file.write(f"Accuracy: {accuracy:.4f}\n")
    summary_file.write("Confusion Matrix:\n")
    summary_file.write(f"{conf_matrix}\n")

print(f"""
=== Machine Learning: Saving Training Report ===
Saved file: {MODEL_TRAINING_SUMMARY_PATH.relative_to(PROJECT_ROOT)}
""")
