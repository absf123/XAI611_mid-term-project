import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    correct = (out == label).sum().item() / total
    return correct

def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label == 1.)
    sens = np.sum(out[mask]) / np.sum(mask)
    return sens

def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total
    return spec

def get_clf_eval(y_test, pred):
    """
    :param y_test, pred: requring dnarray type
    :return:
    """
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_score = roc_auc_score(y_test, pred)
    return accuracy, precision, recall, f1, roc_score
