import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_predictions(labels, predictions, probabilities):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    class_count = probabilities.shape[1]
    one_hot_labels = label_binarize(labels, classes=range(class_count))
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted"),
        "kappa": cohen_kappa_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
        "rmse": np.sqrt(mean_squared_error(one_hot_labels, probabilities)),
        "mae": mean_absolute_error(one_hot_labels, probabilities),
    }
