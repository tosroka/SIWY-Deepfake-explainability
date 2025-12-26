# %%
from sklearn.metrics import classification_report
from typing import Any, Dict, List
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys

# Resolves the issue with finding the utils scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import get_split


def get_all_files(embedding, folders):
    """
    Loads all npy embeddings and their file paths from the given dataset and folders.
    Returns:
        X_sample: np.ndarray of embeddings
        sample_files: list of file paths
    """
    X_sample = []
    sample_files = []
    for folder in folders:
        folder_path = os.path.join(
            "data", folder, "audio", "embeddings", embedding
        )
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                X_sample.append(np.load(os.path.join(folder_path, file)))
                sample_files.append(os.path.join(folder_path, file))
    X_sample = np.array(X_sample)
    return X_sample, sample_files


def load_ircamplify_results(folders):
    true_class = []
    files = []
    is_ai = []
    confidence = []
    for folder in folders:
        folder_path = f"data/ircamplify_results/{folder}"
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), "r") as f:
                    data = json.load(f)
                    job_infos = data.get("job_infos", {})
                    file_paths = job_infos.get("file_paths", {})
                    report_info = job_infos.get("report_info", {})
                    report = report_info.get("report", {})
                    result_list = report.get("resultList", [])

                    for i, result in enumerate(result_list):
                        true_class.append(folder)
                        file = file_paths[i].split("/")[-1]
                        files.append(file)
                        is_ai.append(result.get("isAi"))
                        confidence.append(result.get("confidence"))
    # make it into a dataframe
    data = {
        "true_class": true_class,
        "file": files,
        "is_ai": is_ai,
        "confidence": confidence,
    }
    data = pd.DataFrame(data)
    return data


def get_classifiers_results(models, X_sample_scaled, sample_files):
    true_class = []
    files = []
    svm_pred_parent = []
    svm_pred_child = []
    rf_pred_parent = []
    rf_pred_child = []
    knn_pred_parent = []
    knn_pred_child = []

    for i, file in enumerate(sample_files):
        true_class.append(file.split("/")[-5])
        files.append(file.split("/")[-1].replace("npy", "mp3"))
    for name, model in models.items():
        y_pred = model.predict(X_sample_scaled)
        for i, file in enumerate(sample_files):
            if name == "svc":
                svm_pred_parent.append(y_pred[i, 0])
                svm_pred_child.append(y_pred[i, 1])
            elif name == "rf":
                rf_pred_parent.append(y_pred[i, 0])
                rf_pred_child.append(y_pred[i, 1])
            elif name == "knn":
                knn_pred_parent.append(y_pred[i, 0])
                knn_pred_child.append(y_pred[i, 1])

    data = {
        "true_class": true_class,
        "file": files,
        "svm_pred_parent": svm_pred_parent,
        "svm_pred_child": svm_pred_child,
        "rf_pred_parent": rf_pred_parent,
        "rf_pred_child": rf_pred_child,
        "knn_pred_parent": knn_pred_parent,
        "knn_pred_child": knn_pred_child,
    }
    data = pd.DataFrame(data)
    return data


def get_results_all(folders=["suno", "udio", "lastfm"]):
    # Load trained models and scaler
    with open("artifacts/models_and_scaler.pkl", "rb") as f:
        saved_data = pickle.load(f)
    models = saved_data["models"]
    scaler = saved_data["scaler"]

    # Load sample data
    if "boomy" in folders:
        without_boomy = [folder for folder in folders if folder != "boomy"]
        X_sample, sample_files = get_split(
            "sample", "clap-laion-music", without_boomy
        )
        X_boomy, y_boomy, sample_files_boomy = get_split(
            "sample", "clap-laion-music", ["boomy"]
        )
        X_sample = np.concatenate((X_sample, X_boomy))
        sample_files = sample_files + sample_files_boomy
    else:
        # X_sample, sample_files = get_split('sample', 'clap-laion-music', folders)
        X_sample, sample_files = get_all_files(
            "clap-laion-music", folders
        )  # get all npy emneddings from `audio` subdir

    X_sample_scaled = scaler.transform(X_sample)

    # classifier results
    classifiers_results = get_classifiers_results(
        models, X_sample_scaled, sample_files
    )

    #  how many rows have 'suno', 'udio', 'lastfm' as the true class
    # print(classifiers_results['true_class'].value_counts())
    # if there are ircamplify results, compare
    if os.path.exists("/data/ircamplify_results/"):
        # ircamplify results
        ircamplify_results = load_ircamplify_results(folders)
        # remove rows that have repeated files in ircamplify results
        ircamplify_results = ircamplify_results.drop_duplicates(
            subset="file", keep="first"
        )
        merged_data = pd.merge(
            classifiers_results,
            ircamplify_results,
            on=["true_class", "file"],
            how="left",
        )
        print("length of merged data:", len(merged_data))
        return merged_data
    else:
        return classifiers_results


def print_and_save_classification_report_latex(
    data: pd.DataFrame,
    folders: List[str],
    output_filename: str = "classification_report.tex",
):
    latex_output = []

    y_true = data["true_class"]
    y_pred_svm_parent = data["svm_pred_parent"]
    y_pred_rf_parent = data["rf_pred_parent"]
    y_pred_knn_parent = data["knn_pred_parent"]

    # Binary conversion (AI vs non-AI)
    y_true_ai = np.array([label != "lastfm" for label in y_true])
    y_pred_svm_ai = np.array([label == "AI" for label in y_pred_svm_parent])
    y_pred_rf_ai = np.array([label == "AI" for label in y_pred_rf_parent])
    y_pred_knn_ai = np.array([label == "AI" for label in y_pred_knn_parent])

    classifiers = [
        ("SVM Classifier", y_pred_svm_ai),
        ("RF Classifier", y_pred_rf_ai),
        ("KNN Classifier", y_pred_knn_ai),
    ]

    y_pred_ai = data.get("is_ai")
    if y_pred_ai is not None:
        classifiers.append(("Ircam Amplify Classifier", y_pred_ai))

    # Generate Confusion Matrices
    for title, y_pred in classifiers:
        cm = pd.crosstab(
            y_true_ai, y_pred, rownames=["True"], colnames=["Predicted"]
        )

        # Normalizing across rows
        cm_sum = cm.sum(axis=1).replace(0, 1)
        cm_normalized = cm.div(cm_sum, axis=0)

        caption = f"Normalized Confusion Matrix for {title} (AI vs non-AI)"
        latex_output.append(
            f"\\begin{{table}}[ht]\\centering\n"
            f"{cm_normalized.to_latex(float_format='%.3f')}"
            f"\\caption{{{caption}}}\n"
            f"\\end{{table}}\n"
        )

    # Generate Parent-Level Reports
    svm_report = classification_report(
        y_true_ai, y_pred_svm_ai, output_dict=True
    )
    rf_report = classification_report(
        y_true_ai, y_pred_rf_ai, output_dict=True
    )
    knn_report = classification_report(
        y_true_ai, y_pred_knn_ai, output_dict=True
    )

    classifiers_report = [
        ("SVM", svm_report),
        ("RF", rf_report),
        ("KNN", knn_report),
    ]

    if y_pred_ai is not None:
        ai_report = classification_report(
            y_true_ai, y_pred_ai, output_dict=True
        )
        classifiers_report.append(("Ircam Amp.", ai_report))

    # Build Parent Table
    table = (
        r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
        + "\n"
        + r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
        + "\n"
    )

    for classifier, report in classifiers_report:
        if not isinstance(report, dict):
            continue

        pos_key: str | None = None
        if "True" in report:
            pos_key = "True"
        elif "1" in report:
            pos_key = "1"

        if pos_key is not None:
            metrics = report[pos_key]
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1-score"]
        else:
            precision = recall = f1 = 0.0

        accuracy = report.get("accuracy", 0.0)
        table += (
            f"{classifier} & {precision:.3f} & {recall:.3f} & "
            f"{f1:.3f} & {accuracy:.3f} \\\\ \hline \n"
        )

    table += (
        r"\end{tabular}\caption{Parent-level classification results "
        r"(AI vs. non-AI)}\label{tab:parent_results}\end{table}"
    )
    latex_output.append(table)

    # Child-Level Classification (LastFM, Suno, Udio)
    y_pred_svm_child = data["svm_pred_child"]
    y_pred_rf_child = data["rf_pred_child"]
    y_pred_knn_child = data["knn_pred_child"]

    child_reports = [
        (
            "SVM",
            classification_report(y_true, y_pred_svm_child, output_dict=True),
        ),
        (
            "RF",
            classification_report(y_true, y_pred_rf_child, output_dict=True),
        ),
        (
            "KNN",
            classification_report(y_true, y_pred_knn_child, output_dict=True),
        ),
    ]

    # Macro-Averaged Table
    table_macro = (
        r"\begin{table}[ht]\centering\begin{tabular}{lcccc}\hline"
        + "\n"
        + r"Classifier & Precision & Recall & F1-Score & Accuracy \\ \hline"
        + "\n"
    )

    for classifier, report in child_reports:
        if not isinstance(report, dict):
            continue

        m_avg = report.get("macro avg", {})
        prec = m_avg.get("precision", 0.0)
        rec = m_avg.get("recall", 0.0)
        f1 = m_avg.get("f1-score", 0.0)
        acc = report.get("accuracy", 0.0)
        table_macro += (
            f"{classifier} & {prec:.3f} & {rec:.3f} & "
            f"{f1:.3f} & {acc:.3f} \\\\ \hline \n"
        )

    table_macro += (
        r"\end{tabular}\caption{Child-level results (Macro-Averaged)}"
        r"\label{tab:child_macro}\end{table}"
    )
    latex_output.append(table_macro)

    # Detailed Child-Level Table
    detailed_table = (
        r"\begin{table}[ht]\centering\begin{tabular}{llccc}\hline"
        + "\n"
        + r"Classifier & Category & Precision & Recall & F1-score \\ \hline"
        + "\n"
    )

    for classifier, report in child_reports:
        if not isinstance(report, dict):
            continue

        num_cats = len(folders)
        detailed_table += f"\\multirow{{{num_cats}}}{{*}}{{{classifier}}} & "

        for idx, category in enumerate(folders):
            cat_data = report.get(
                category, {"precision": 0, "recall": 0, "f1-score": 0}
            )

            p = cat_data["precision"]
            r = cat_data["recall"]
            f = cat_data["f1-score"]

            row_prefix = "& " if idx > 0 else ""
            detailed_table += (
                f"{row_prefix}{category.capitalize()} & "
                f"{p:.3f} & {r:.3f} & {f:.3f} \\\\ \n"
            )
        detailed_table += r"\hline " + "\n"

    detailed_table += (
        r"\end{tabular}\caption{Detailed child-level classification results}"
        r"\label{tab:child_detailed}\end{table}"
    )
    latex_output.append(detailed_table)

    # Save report to file
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_output))

        print(f"Report saved to: {os.path.abspath(output_filename)}")
    except IOError as e:
        print(f"Error saving report: {e}")


# Example usage
if __name__ == "__main__":
    with_boomy = False

    if with_boomy:
        folders = ["suno", "udio", "lastfm", "boomy"]
    else:
        folders = ["suno", "udio", "lastfm"]

    data = get_results_all(folders)
    print_and_save_classification_report_latex(data, folders)
