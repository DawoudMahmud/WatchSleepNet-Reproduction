"""
Evaluation functions for WatchSleepNet reproduction.
Implements accuracy, F1-score, Cohen’s kappa, and confusion matrix computation.
"""

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

def evaluate_model(y_true, y_pred, average="macro"):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average=average),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    return results
