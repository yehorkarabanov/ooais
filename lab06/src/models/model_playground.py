from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

PROCESSED_DATA_PATH = DATA_PATH / "processed"

MODEL_FEATURES_PATH = PROCESSED_DATA_PATH / "model_features.csv"
MODEL_LABELS_PATH = PROCESSED_DATA_PATH / "model_labels.csv"

RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

REPORTS_PATH: Path = PROJECT_ROOT / "reports"
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_TRAINING_SUMMARY_PATH = REPORTS_PATH / "model_training_summary.txt"
MODEL_PLAYGROUND_SUMMARY_PATH = REPORTS_PATH / "model_playground_summary.txt"
MODEL_COMPARISON_PLOT_PATH = REPORTS_PATH / "model_comparison_panel.png"

required_input_dirs = [DATA_PATH, PROCESSED_DATA_PATH]
missing_dirs = [path for path in required_input_dirs if not path.exists()]
if missing_dirs:
    print("Error: missing required input directory(ies):")
    for path in missing_dirs:
        print(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")
    raise SystemExit(1)

required_input_files = [MODEL_FEATURES_PATH, MODEL_LABELS_PATH]
missing_files = [path for path in required_input_files if not path.exists()]
if missing_files:
    print("Error: missing required input file(s):")
    for path in missing_files:
        print(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")
    raise SystemExit(1)


def load_data():
    features_df = pd.read_csv(MODEL_FEATURES_PATH)
    labels_df = pd.read_csv(MODEL_LABELS_PATH)
    print(f"""
=== Model Playground: Loading Data ===
Feature file: {MODEL_FEATURES_PATH.relative_to(PROJECT_ROOT)}
Label file: {MODEL_LABELS_PATH.relative_to(PROJECT_ROOT)}
    """)
    return features_df, labels_df


def inspect_data(features_df, labels_df):
    if features_df.empty or labels_df.empty:
        print("Error: empty input file(s):")
        raise SystemExit(1)

    if len(features_df) != len(labels_df):
        print("Error: inconsistent input file(s):")
        raise SystemExit(1)

    if len(features_df) == 0:
        print("Error: empty input file(s):")
        raise SystemExit(1)

    if len(labels_df) == 0:
        print("Error: empty input file(s):")
        raise SystemExit(1)

    if features_df.shape[0] != labels_df.shape[0]:
        print("Error: inconsistent input file(s):")
        raise SystemExit(1)

    if "anomaly_flag" not in labels_df.columns:
        print("Error: missing required column 'anomaly_flag' in labels file")
        raise SystemExit(1)

    print(f"""
=== Model Playground: Data Inspection ===
Number of samples: {len(features_df)}
Number of features: {len(features_df.columns)}
Feature columns: {list(features_df.columns)}
Target values detected: {labels_df["anomaly_flag"].unique()}
    """)


def prepare_features_and_labels(f_df, l_df):
    x = f_df.values
    y = l_df["anomaly_flag"].astype(int).values
    print(f"""
=== Model Playground: Preparing Features and Labels ===
X shape: {x.shape}
y shape: {y.shape}
""")
    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(f"""
=== Model Playground: Train/Test Split ===
Training samples: {len(x_train)}
Testing samples: {len(x_test)}
""")
    # Keep the return order aligned with unpacking in __main__.
    return x_train, x_test, y_train, y_test


def define_models():
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    return models


def train_models(models, x_train, y_train):
    trained_models = {}
    print("\n=== Model Playground: Training Models ===\n")
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        print(f"{model_name}: trained")
        trained_models[model_name] = model

    return trained_models


def generate_predictions(trained_models, x_test):
    results = []
    for model_name, model in trained_models.items():
        predictions = model.predict(x_test)
        print(f"{model_name}: generated predictions for test set")
        result = {
            "name": model_name,
            "model": model,
            "y_pred": predictions,
        }
        results.append(result)
    return results


def print_example_predictions(prediction_results, y_test, num_examples=5):
    print("\n=== Model Playground: Example Predictions ===\n")
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
        print(line)


def compute_accuracy(prediction_results, y_test):
    print("\n=== Model Playground: Accuracy Comparison ===\n")
    for result in prediction_results:
        y_pred = result["y_pred"]
        accuracy = accuracy_score(y_test, y_pred)
        result["accuracy"] = accuracy
        print(f"{result['name']}: {accuracy:.4f}")

    return prediction_results


def compute_detailed_metrics(prediction_results, y_test):
    print("\n=== Model Playground: Detailed Metrics ===\n")

    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        result["confusion_matrix"] = cm
        result["classification_report"] = report

        print(f"Model: {result['name']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        print("\nClassification Report:")
        print("--------------------------------------------------------------------")
        print(
            f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}"
        )
        print("--------------------------------------------------------------------")
        print(
            f"{'0 (normal)':<16}"
            f"{report['0']['precision']:>10.2f}"
            f"{report['0']['recall']:>10.2f}"
            f"{report['0']['f1-score']:>10.2f}"
            f"{int(report['0']['support']):>10}"
        )
        print(
            f"{'1 (anomaly)':<16}"
            f"{report['1']['precision']:>10.2f}"
            f"{report['1']['recall']:>10.2f}"
            f"{report['1']['f1-score']:>10.2f}"
            f"{int(report['1']['support']):>10}"
        )
        print("--------------------------------------------------------------------")
        print(
            f"{'Macro average':<16}"
            f"{report['macro avg']['precision']:>10.2f}"
            f"{report['macro avg']['recall']:>10.2f}"
            f"{report['macro avg']['f1-score']:>10.2f}"
            f"{int(report['macro avg']['support']):>10}"
        )
        print(
            f"{'Weighted average':<16}"
            f"{report['weighted avg']['precision']:>10.2f}"
            f"{report['weighted avg']['recall']:>10.2f}"
            f"{report['weighted avg']['f1-score']:>10.2f}"
            f"{int(report['weighted avg']['support']):>10}"
        )
        print()

    return prediction_results


def rank_models(evaluation_results):
    print("=== Model Playground: Ranking ===")
    sorted_results = sorted(
        evaluation_results, key=lambda result: result["accuracy"], reverse=True
    )
    for index, result in enumerate(sorted_results, start=1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")

    return sorted_results


def save_experiment_summary(
    features_path,
    labels_path,
    x,
    x_train,
    x_test,
    ranked_models,
    experiment_results,
):
    print("\n=== Model Playground: Saving Summary ===")

    num_samples = x.shape[0]
    num_features = x.shape[1]
    best_model = ranked_models[0] if ranked_models else None

    with MODEL_PLAYGROUND_SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        summary_file.write("OOAIS Model Playground Summary\n")
        summary_file.write("================================\n\n")

        summary_file.write("Input datasets\n")
        summary_file.write("--------------\n")
        summary_file.write(f"{Path(features_path).relative_to(PROJECT_ROOT).as_posix()}\n")
        summary_file.write(f"{Path(labels_path).relative_to(PROJECT_ROOT).as_posix()}\n\n")

        summary_file.write("Dataset statistics\n")
        summary_file.write("------------------\n")
        summary_file.write(f"Number of samples: {num_samples}\n")
        summary_file.write(f"Number of features: {num_features}\n")
        summary_file.write(f"Training samples: {len(x_train)}\n")
        summary_file.write(f"Testing samples: {len(x_test)}\n\n")

        summary_file.write("Compared models\n")
        summary_file.write("---------------\n")
        for result in ranked_models:
            summary_file.write(f"- {result['name']}: {result['accuracy']:.4f}\n")
        summary_file.write("\n")

        summary_file.write("Best model\n")
        summary_file.write("----------\n")
        if best_model is None:
            summary_file.write("No models were evaluated.\n\n")
        else:
            summary_file.write(
                f"{best_model['name']} achieved the highest accuracy: "
                f"{best_model['accuracy']:.4f}\n\n"
            )

        summary_file.write("Selected detailed evaluation\n")
        summary_file.write("----------------------------\n")
        if best_model is not None:
            best_report = best_model.get("classification_report", {})
            anomaly_metrics = best_report.get("1", {})
            summary_file.write(f"Model: {best_model['name']}\n")
            summary_file.write("Confusion matrix:\n")
            summary_file.write(f"{best_model.get('confusion_matrix')}\n")
            summary_file.write(
                "Anomaly class metrics (1): "
                f"precision={anomaly_metrics.get('precision', 0.0):.4f}, "
                f"recall={anomaly_metrics.get('recall', 0.0):.4f}, "
                f"f1-score={anomaly_metrics.get('f1-score', 0.0):.4f}\n\n"
            )
        else:
            summary_file.write("Detailed metrics are unavailable.\n\n")

        summary_file.write("Additional experiments\n")
        summary_file.write("----------------------\n")
        for exp in experiment_results:
            summary_file.write(f"- {exp['name']}: {exp['accuracy']:.4f}\n")
        summary_file.write("\n")

        summary_file.write("Conclusion\n")
        summary_file.write("----------\n")
        if best_model is None:
            summary_file.write(
                "No final recommendation is available because no model results were produced.\n"
            )
        else:
            summary_file.write(
                "The best candidate for further experiments is "
                f"{best_model['name']}, because it achieved the highest accuracy "
                "on the current test set.\n"
            )

    print(f"Saved file: {MODEL_PLAYGROUND_SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}")


def create_metric_plots(ranked_models):
    print("\n=== Model Playground: Saving Visualizations ===")

    model_names = [str(result["name"]) for result in ranked_models]
    accuracies = [result["accuracy"] for result in ranked_models]
    precisions = [
        result["classification_report"].get("1", {}).get("precision", 0.0)
        for result in ranked_models
    ]
    recalls = [
        result["classification_report"].get("1", {}).get("recall", 0.0)
        for result in ranked_models
    ]
    f1_scores = [
        result["classification_report"].get("1", {}).get("f1-score", 0.0)
        for result in ranked_models
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title("Accuracy")
    axes[0, 1].bar(model_names, precisions)
    axes[0, 1].set_title("Precision (Anomaly)")
    axes[1, 0].bar(model_names, recalls)
    axes[1, 0].set_title("Recall (Anomaly)")
    axes[1, 1].bar(model_names, f1_scores)
    axes[1, 1].set_title("F1-score (Anomaly)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PLOT_PATH)
    plt.close(fig)

    print(f"Saved file: {MODEL_COMPARISON_PLOT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


def pipeline(x_train, x_test, y_train, y_test, models):
    trained_model = train_models(models, x_train, y_train)
    results = generate_predictions(trained_model, x_test)
    print_example_predictions(results, y_test)

    prediction_results = compute_accuracy(results, y_test)
    evaluation_results = compute_detailed_metrics(prediction_results, y_test)
    sorted_results = rank_models(evaluation_results)
    return sorted_results

if __name__ == "__main__":
    f_df, l_df = load_data()
    inspect_data(f_df, l_df)
    x, y = prepare_features_and_labels(f_df, l_df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    models = define_models()
    sorted_results = pipeline(x_train, x_test, y_train, y_test, models)

    # ex 1
    # models = {
    #     "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
    #     "Logistic Regression": LogisticRegression(max_iter=1000),
    #     "Random Forest": RandomForestClassifier(random_state=42),
    # }
    ex1_models = {}
    ex2_models = {}
    depths = [2, 3, 5]
    for depth in depths:
        ex1_models[depth] = DecisionTreeClassifier(max_depth=depth, random_state=42)

    n_estimatorss = [5, 10, 50]
    for n_estimators in n_estimatorss:
        ex2_models[n_estimators] = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42
        )

    sorted_ex1_results = pipeline(x_train, x_test, y_train, y_test, ex1_models)
    sorted_ex2_results = pipeline(x_train, x_test, y_train, y_test, ex2_models)

    ex1_accuracy_by_depth = {
        result["name"]: result["accuracy"] for result in sorted_ex1_results
    }
    ex2_accuracy_by_trees = {
        result["name"]: result["accuracy"] for result in sorted_ex2_results
    }

    ex1_accuracies = [ex1_accuracy_by_depth[depth] for depth in depths]
    ex2_accuracies = [ex2_accuracy_by_trees[n_estimators] for n_estimators in n_estimatorss]

    plt.plot(depths, ex1_accuracies, "b-", label="Accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree performance vs max_depth")
    plt.show()

    plt.plot(n_estimatorss, ex2_accuracies, "r-", label="Accuracy")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy")
    plt.title("Random Forest performance vs n_estimators")
    plt.show()

    experiment_results = [
        {
            "name": f"Decision Tree (max_depth={depth})",
            "accuracy": ex1_accuracy_by_depth[depth],
        }
        for depth in depths
    ]
    experiment_results.extend(
        {
            "name": f"Random Forest (n_estimators={n_estimators})",
            "accuracy": ex2_accuracy_by_trees[n_estimators],
        }
        for n_estimators in n_estimatorss
    )

    save_experiment_summary(
        MODEL_FEATURES_PATH,
        MODEL_LABELS_PATH,
        x,
        x_train,
        x_test,
        sorted_results,
        experiment_results,
    )
    create_metric_plots(sorted_results)

