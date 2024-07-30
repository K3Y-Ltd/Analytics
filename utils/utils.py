from typing import List, Dict
import numpy as np
import sklearn.metrics as metrics
import requests
import os
from joblib import load

from sklearn.metrics import mean_absolute_error
from scipy.stats import wasserstein_distance
def affinity(matrix1, matrix2):
    """
    Compute the column-wise correlation between two data matrices.

    Parameters:
        matrix1 (numpy.ndarray): First data matrix.
        matrix2 (numpy.ndarray): Second data matrix.

    Returns:
        numpy.ndarray: Array containing the correlation coefficients for each pair of columns.
    """
    # Ensure matrices have the same number of columns
    assert matrix1.shape[1] == matrix2.shape[1], "Matrices must have the same number of columns"

    # Initialize an array to store correlation coefficients
    correlations = np.zeros(matrix1.shape[1])
    w_distances = np.zeros(matrix1.shape[1])

    # Compute correlation coefficient for each pair of columns
    for i in range(matrix1.shape[1]):
        correlations[i] = np.corrcoef(matrix1[:, i], matrix2[:, i])[0, 1]
        w_distances[i] = wasserstein_distance(matrix1[:, i], matrix2[:, i])
    diff = mean_absolute_error(matrix1, matrix2)

    return correlations[~np.isnan(correlations)].mean(), diff, w_distances[~np.isnan(w_distances)].mean()


def download_file(url: str, save_path: str, file_name: str) -> any:
    """
    Downloads a file from a specified URL.
    Args:
        url (str): The URL from which the file will be downloaded.
        save_path (str): The local directory path where the file will be saved.
        file_name (str): The name to be used for saving the downloaded file.
    Returns:
        None: Prints download success or error messages.
    """
    # Make the GET request to download the file
    print(url + '/files/' + file_name)
    response = requests.get(url + '/files/' + file_name)

    if response.status_code == 200:

        file_path = os.path.join(save_path, file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File [{file_name}] download successful. Saved at: {file_path}")
        return load(os.path.join(save_path, file_name))
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def false_positive_rate(y_test: List[int], y_probs: np.ndarray, class_mapping: Dict[int, str]) -> float:
    """
    Calculate the mean false positive rate for a multi-class classification problem.

    Args:
        y_test (List[int]): True labels.
        y_probs (np.ndarray): Predicted probabilities for each class.
        class_mapping (Dict[int, str]): Mapping of class indices to class names.

    Returns:
        float: Mean false positive rate.
    """
    y_preds = y_probs.argmax(1)
    y_test = [class_mapping[i] for i in y_test]
    confusion_mat = metrics.confusion_matrix(y_test, y_preds)

    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)
    fpr = FP / (FP+TN)

    return fpr.mean()

def true_positive_rate(y_test, y_probs, class_mapping):
    """
    Calculate the mean true positive rate for a multi-class classification problem.

    Args:
        y_test (List[int]): True labels.
        y_probs (np.ndarray): Predicted probabilities for each class.
        class_mapping (Dict[int, str]): Mapping of class indices to class names.

    Returns:
        float: Mean true positive rate.
    """
    y_preds = y_probs.argmax(1)
    y_test = [class_mapping[i] for i in y_test]
    confusion_mat = metrics.confusion_matrix(y_test, y_preds)

    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    tpr = TP / (TP+FN)

    return tpr.mean()

def accuracy_score(y_test: List[int], y_probs: np.ndarray, class_mapping: Dict[int, str]) -> float:
    """
    Calculate the accuracy score for a multi-class classification problem.

    Args:
        y_test (List[int]): True labels.
        y_probs (np.ndarray): Predicted probabilities for each class.
        class_mapping (Dict[int, str]): Mapping of class indices to class names.

    Returns:
        float: Accuracy score.
    """
    y_preds = y_probs.argmax(1)
    y_test = [class_mapping[i] for i in y_test]
    return metrics.accuracy_score(y_test, y_preds)

def f1_score(y_test: List[int], y_probs: np.ndarray, class_mapping: Dict[int, str]) -> float:
    """
    Calculate the weighted F1 score for a multi-class classification problem.

    Args:
        y_test (List[int]): True labels.
        y_probs (np.ndarray): Predicted probabilities for each class.
        class_mapping (Dict[int, str]): Mapping of class indices to class names.

    Returns:
        float: Weighted F1 score.
    """
    y_preds = y_probs.argmax(1)
    y_test = [class_mapping[i] for i in y_test]
    return metrics.f1_score(y_test, y_preds, average='weighted')


def roc_auc(y_test: List[int], y_probs: np.ndarray, class_mapping: Dict[int, str]) -> float:
    """
    Calculate the weighted ROC AUC score for a multi-class classification problem.

    Parameters:
        y_test (List[int]): True labels.
        y_probs (np.ndarray): Predicted probabilities for each class.
        class_mapping (Dict[int, str]): Mapping of class indices to class names.

    Returns:
        float: Weighted ROC AUC score.
    """
    y_test = [class_mapping[i] for i in y_test]

    # if there is nan in some row, replace with uniform prob
    if np.any(np.isnan(y_probs), axis=1).any():
        nan_indices = np.any(np.isnan(y_probs), axis=1)
        y_probs[nan_indices,:] = np.full(y_probs.shape[1], 1.0 / y_probs.shape[1])

    # metrics.roc_auc_score behaves differently in the binary and multi-class scenarios
    if len(set(y_test)) > 2:
        return metrics.roc_auc_score(y_test, y_probs, average='weighted', multi_class='ovr')
    else:
        return metrics.roc_auc_score(y_test, y_probs[:,1], average='weighted')
