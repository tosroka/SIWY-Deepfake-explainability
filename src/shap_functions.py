
import pickle
import numpy as np
from sklearn.svm._classes import SVC
import shap
from pathlib import Path
import matplotlib.pyplot as plt
shap.initjs()

def _inspect_SVC(model, X_background: np.ndarray, X_test, kmeans, decision_fun):
    outpath = Path(f"shap_values_{decision_fun.__qualname__}_{"kmeans" if kmeans else "50"}.npy")
    print("Outpath is",outpath)
    if kmeans:
        X_background_explainer = shap.kmeans(X_background, k=50)
        print("kmeans size",X_background_explainer.data.shape)
    else:
        X_background_explainer = X_background[:50]
    print("Creating explainer")
    explainer = shap.KernelExplainer(
        decision_fun,
        X_background_explainer  # ~50 representative samples
    )

    if not outpath.exists():
        shap_values = explainer.shap_values(X_test)
        np.save(outpath, shap_values)
    else:
        shap_values = np.load(outpath)
    print(shap_values, shap_values.shape)

    return shap_values


def inspect_SVC(model, X_background, X_test, kmeans=False):
    shap_values = _inspect_SVC(model, X_background, X_test, kmeans, model.decision_function)

    plt.title("SHAP values for decision boundary distance")
    shap.summary_plot(shap_values, X_test, plot_type="bar")
def inspect_SVC_proba(model, X_background, X_test, kmeans=False):
    print("X_background shape:",X_background.shape)
    print("X_test shape:",X_test.shape)
    shap_values = _inspect_SVC(model, X_background, X_test, kmeans, model.predict_proba)

    print("SHAP values shape:",shap_values.shape)

    plt.title("SHAP values for AI class")
    shap.summary_plot(shap_values[:,:,0], X_test)
    plt.show()
    plt.savefig("figures/svc/shap_svc_proba_summary_ai.png")

    plt.title("SHAP values for Non-AI class")

    shap.summary_plot(shap_values[:,:,1], X_test)
    plt.show()
    plt.savefig("figures/svc/shap_svc_proba_summary_nonai.png")
if __name__ == "__main__":
    feature_names = None
    with open('models_and_scaler.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    models = saved_data['models']
    scaler = saved_data['scaler']
    feature_names = None
    AI_classifier: SVC = models["svc"].hierarchy_.nodes["AI"]["classifier"]
    positive_index = np.where(AI_classifier.classes_ == 1)[0]