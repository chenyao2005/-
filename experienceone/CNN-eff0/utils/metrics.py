from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
    }


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, target_names=None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0, digits=4)
