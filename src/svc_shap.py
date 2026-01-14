# %%
import logging
import numpy as np
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

logging.basicConfig()

import sys, os

from sklearn.svm import SVC
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shap_functions import inspect_SVC, inspect_SVC_proba
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
        folder_path = os.path.join('data', folder, 'audio', 'embeddings', embedding)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.npy'):
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
        folder_path = f'data/ircamplify_results/{folder}'
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), 'r') as f:
                    data = json.load(f)
                    job_infos = data.get('job_infos', {})
                    file_paths = job_infos.get('file_paths', {})
                    report_info = job_infos.get('report_info', {})
                    report = report_info.get('report', {})
                    result_list = report.get('resultList', [])
                    
                    for i, result in enumerate(result_list):
                        true_class.append(folder)
                        file = file_paths[i].split('/')[-1]
                        files.append(file)
                        is_ai.append(result.get('isAi'))
                        confidence.append(result.get('confidence'))
    # make it into a dataframe
    data = {
        'true_class': true_class,
        'file': files,
        'is_ai': is_ai,
        'confidence': confidence
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
        true_class.append(file.split('/')[-5])
        files.append(file.split('/')[-1].replace('npy','mp3'))
    for name, model in models.items():
        y_pred = model.predict(X_sample_scaled)
        for i, file in enumerate(sample_files):
            if name == 'svc':
                svm_pred_parent.append(y_pred[i, 0])
                svm_pred_child.append(y_pred[i, 1])
            elif name == 'rf':
                rf_pred_parent.append(y_pred[i, 0])
                rf_pred_child.append(y_pred[i, 1])
            elif name == 'knn':
                knn_pred_parent.append(y_pred[i, 0])
                knn_pred_child.append(y_pred[i, 1])


    data = {
        'true_class': true_class,
        'file': files,
        'svm_pred_parent': svm_pred_parent,
        'svm_pred_child': svm_pred_child,
        'rf_pred_parent': rf_pred_parent,
        'rf_pred_child': rf_pred_child,
        'knn_pred_parent': knn_pred_parent,
        'knn_pred_child': knn_pred_child
    }
    data = pd.DataFrame(data)
    return data

def get_model_and_scaled_samples():
    # Load trained models and scaler
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    models = saved_data['models']
    scaler = saved_data['scaler']

    # Load sample data
    if 'boomy' in folders:
        without_boomy = [folder for folder in folders if folder != 'boomy']
        X_sample, sample_files = get_split('sample', 'clap-laion-music', without_boomy)
        #X_boomy, y_boomy, sample_files_boomy = get_split('sample', 'clap-laion-music', ['boomy'])
        #X_sample = np.concatenate((X_sample, X_boomy))
        #sample_files = sample_files + sample_files_boomy
    else:
        # X_sample, sample_files = get_split('sample', 'clap-laion-music', folders)
        X_sample, sample_files = get_all_files('clap-laion-music', folders) # get all npy emneddings from `audio` subdir
    
    X_sample_scaled = scaler.transform(X_sample)

    return X_sample_scaled, sample_files, models


def get_results_all(folders, X_sample_scaled, sample_files, models):   
    # classifier results
    classifiers_results = get_classifiers_results(models, X_sample_scaled, sample_files)

    #  how many rows have 'suno', 'udio', 'lastfm' as the true class
    # print(classifiers_results['true_class'].value_counts())
    # if there are ircamplify results, compare
    if os.path.exists('/data/ircamplify_results/'):
        # ircamplify results
        ircamplify_results = load_ircamplify_results(folders)
        # remove rows that have repeated files in ircamplify results
        ircamplify_results = ircamplify_results.drop_duplicates(subset='file', keep='first')
        merged_data = pd.merge(classifiers_results, ircamplify_results, on=['true_class', 'file'], how='left')
        print('length of merged data:', len(merged_data))
        return merged_data
    else:
        return classifiers_results

# Example usage
obstruction_test = False
kmeans = True # get average of training samples

if __name__ == "__main__":
    log = logging.getLogger("XAI")
    log.setLevel(logging.INFO)
    with_boomy = False

    if with_boomy:
        folders = ['suno', 'udio', 'lastfm', 'boomy']
    else:
        folders = ['suno', 'udio', 'lastfm']

    log.info("Loading models and samples")
    X_sample_scaled, sample_files, models= get_model_and_scaled_samples()
    a = -5
    # what happens if we replace important features?
    # answer: nothing, or the sample becomes heavily out of distribution and breaks the model
    if obstruction_test:
        X_sample_scaled[:,356] = a
        X_sample_scaled[:,399] = a
    log.info("Evaluating")
    data = get_results_all(folders, X_sample_scaled, sample_files, models)
    # first in hierarchy
    AI_classifier: SVC = models["svc"].hierarchy_.nodes["AI"]["classifier"]
    positive_index = np.where(AI_classifier.classes_ == 1)[0]

    #print(data)

    mapping = {
        'suno': 'AI',
        'udio': 'AI',
        'lastfm': 'nonAI'
    }

    # get confusion matrix from data
    data['true_parent'] = data['true_class'].map(mapping)
    y_true = data['true_parent']
    y_pred = data['svm_pred_parent']

    #print(classification_report(y_true, y_pred, output_dict=True))

    log.info("Inspecting SVC")

    inspect_SVC_proba(AI_classifier, X_sample_scaled[:-10][:50], X_sample_scaled[-10:], kmeans)

    #print_classification_report_latex(data, folders)
