from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_proba]
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    auc = round(roc_auc_score(y_true, y_pred), 6)

    result = {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1_score": round(f1, 6),
        "auc": auc
    }
    return result
