from lime.lime_tabular import LimeTabularExplainer
import pickle
from sklearn.preprocessing._data import StandardScaler
import numpy as np
import os
import pandas as pd

from sklearn.svm import SVC

from lime.explanation import Explanation
import matplotlib.pyplot as plt

def train_lime(model: SVC, data: np.ndarray, true_class: np.ndarray):
    print("Training lime for svm")
    print("Data shape:",data.shape) # (x,512) embeddings
    print("classes_shape:", true_class.shape)

    train_data = data[:150]
    test_data = data[150:]
    explainer = LimeTabularExplainer(
        training_data=train_data,
        feature_names=[f'feature_{i}' for i in range(train_data.shape[1])],
        class_names=["AI", "not-AI"],
        mode='classification'
    )
    for i in range(5):
        exp: Explanation = explainer.explain_instance(
            data_row=test_data[i],
            predict_fn=model.predict_proba,
        )
        print(f"Explanation for instance {i} (true class: {true_class[i]}):")
        for feature, weight in exp.as_list():
            print(f"  {feature}: {weight}")
        fig = exp.as_pyplot_figure()
        plt.show()
        plt.savefig(f'figures/svm/lime_explanation_instance_{i}.png')


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

def main(folders):
    # load svm and scaler
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    AI_classifier: SVC = saved_data["models"]["svc"].hierarchy_.nodes["AI"]["classifier"]
    scaler: StandardScaler = saved_data['scaler']

    # X_sample, sample_files = get_split('sample', 'clap-laion-music', folders)
    X_sample, sample_files = get_all_files('clap-laion-music', folders) # get all npy emneddings from `audio` subdir

    true_class: list[str] = []
    for i, file in enumerate(sample_files):
        true_class.append(file.split('/')[-5])

    X_sample_scaled = scaler.transform(X_sample)

    mapping = {
        'suno': 'AI',
        'udio': 'AI',
        'lastfm': 'nonAI'
    }

    # get confusion matrix from data
    true_class = [mapping[item] for item in true_class]

    train_lime(AI_classifier, X_sample_scaled, np.array(true_class))

if __name__=="__main__":
    main(['suno', 'udio', 'lastfm'])