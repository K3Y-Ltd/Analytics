import os.path

import requests
import argparse
from typing import Dict, Union, List, Any

import json
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from ..utils.utils import accuracy_score, f1_score, roc_auc, false_positive_rate, true_positive_rate
from ..utils.datasets import process_json_data

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC

from joblib import dump
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from ..utils.datasets import get_json_data
np.seterr(divide='ignore', invalid='ignore')
simplefilter("ignore", category=(ConvergenceWarning))



def train_and_evaluate(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Args:
        model (estimator): The machine learning model to be trained and evaluated.
        X_train (array-like): The feature matrix of the training data.
        y_train (array-like): The target variable of the training data.
        X_test (array-like): The feature matrix of the test data.
        y_test (array-like): The target variable of the test data.
    Returns:
        float: The accuracy of the trained model on the test set.
    """

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set and evaluate
    y_probs = model.predict_proba(X_test) if model.__class__.__name__ not in ['LinearSVC'] else model._predict_proba_lr(X_test)

    # Make the class mapping and evaluate all possible metrics
    class_mapping = {label:label_idx  for label_idx, label in enumerate(model.classes_)}
    print(y_test.shape, y_probs.shape)
    metrics = [metric(y_test, y_probs, class_mapping)
               for metric in [accuracy_score, f1_score, roc_auc, true_positive_rate, false_positive_rate]]

    return metrics


def find_best_config(meta_data: pd.DataFrame, metric: str) -> pd.Series:
    """
    Derives the optimal model configuration based on a specified metric for some class of models. i.e. LR or LDA.
    Args:
        meta_data (pd.DataFrame): DataFrame containing metadata, including metrics.
        metric (str): The metric to consider for choosing the best model configuration.
                   Valid options: "accuracy", "f1_score", "roc_auc", "TPR", "FPR".
    Returns:
        pd.Series: The row corresponding to the optimal model configuration.
    """

    # Construct the metric: idx mapping
    metric_mapping = {"accuracy": 0, "f1_score": 1, "roc_auc": 2, "TPR": 3, "FPR": 4}

    # Check if the specified metric is valid
    if metric not in metric_mapping:
        raise ValueError(f"Invalid metric '{metric}'. Valid options: {', '.join(metric_mapping.keys())}")

    # Choose the best model configuration based on the metric argmax or argmin
    metric_idx = metric_mapping[metric]
    metrics_np = np.array([list(row.values()) for row in meta_data['metrics']])
    best_idx = metrics_np[:, metric_idx].argmax() if metric not in ['FPR'] else metrics_np[:, metric_idx].argmin()
    return meta_data.loc[best_idx]


def cv_customized(X: pd.DataFrame, y: pd.Series, kf: KFold, models: List[BaseEstimator], metric: str) -> tuple[
    Any, Series]:
    """
    Perform cross-validation on a list of models.
    Args:
        X (array-like): Feature matrix of the data.
        y (array-like): Target variable of the data.
        kf (KFold): Cross-validation generator.
        models (list): List of sklearn models to be trained and evaluated.
        metric (str): The metric to choose the best model from.
    Returns:
         tuple: Tuple containing the best model and its corresponding metadata.
    """
    # Initialize metric names
    metric_names = ["accuracy", "f1_score", "roc_auc", "TPR", "FPR"]

    # Perform CV for all linear models. Note that for a large number of models, the training time might grow
    cv_results = np.zeros((len(models), args.folds, 5))

    # Initialize meta-data to store in pd.DataFrame
    column_names = ['name', 'params', 'metrics']
    meta_data = pd.DataFrame(columns=column_names)

    # Iterate over models
    for model_idx, model in enumerate(models):

        print(f"Model {model_idx+1} is being trained.")

        # Iterate over folds
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):

            # Train and evaluate the model for each fold
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
            cv_results[model_idx, fold_idx, :] = fold_metrics

        # Estimate per-model CV performance
        model_cv_estimate = cv_results.mean(axis=1)[model_idx]

        # Store per-model meta-data
        metrics = {name:estimate for name, estimate in zip(metric_names, model_cv_estimate)}
        new_row = {'name': model.__class__.__name__, 'params': model.get_params(), 'metrics': metrics}
        meta_data = meta_data.append(new_row, ignore_index=True)

    # Choose best per-class model (LR, LDA, SVM) based on the metric of interest
    best_config = find_best_config(meta_data, metric)

    # Initialize and train the best per-class model on the whole training set for better performance
    best_model = globals()[best_config["name"]](**best_config["params"]).fit(X, y)

    return best_model, best_config


def model_selection(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Union[BaseEstimator, DataFrame]]]:
    """
    Perform cross-validation on Logistic Regression, Linear Discriminant Analysis, and Support Vector Machines.
    Args:
        X (pd.DataFrame): Feature matrix of the data.
        y (pd.Series): Target variable of the data.
    Returns:
        dict: Dictionary containing the best model and its corresponding metadata for each type of model.
    """
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    # Define a grid of parameters and generate all possible combinations of parameters for Logistic Regression
    print("Training Logistic Regression models:")

    lr_param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 0.5, 1.0, 10.0], 'solver': ['liblinear'], 'max_iter':[100]}
    lr_grid_combinations = ParameterGrid(lr_param_grid)
    lr_models = [LogisticRegression(**params) for params in lr_grid_combinations]
    lr_best, lr_meta = cv_customized(X, y, kf, lr_models, args.metric)

    # Define a grid of parameters and generate all possible combinations of parameters for Discriminant Analysis
    print("Training Linear Discriminant Analysis models:")
    lda_param_grid = {'solver': ['svd', 'lsqr'], 'n_components': [None, 1]}
    lda_grid_combinations = ParameterGrid(lda_param_grid)
    lda_models = [LinearDiscriminantAnalysis(**params) for params in lda_grid_combinations]
    lda_best, lda_meta = cv_customized(X, y, kf, lda_models, args.metric)

    # Define a grid of parameters and generate all possible combinations of parameters for Discriminant Analysis
    print("Training Support Vector models:")
    svm_param_grid = {'C': [0.1, 1, 10, 10], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'max_iter': [300]}
    svm_grid_combinations = ParameterGrid(svm_param_grid)
    svm_models = [LinearSVC(**params) for params in svm_grid_combinations]
    svm_best, svm_meta = cv_customized(X, y, kf, svm_models, args.metric)

    # Structure the output dict with model weights and metadata
    output = {"LR": {"model": lr_best, "metadata": lr_meta},
              "LDA": {"model": lda_best, "metadata": lda_meta},
              "SVM": {"model": svm_best, "metadata": svm_meta}}

    return output


def send_data(url: str, path:str, file_name: str, meta_data: dict) -> None:
    """
    Uploads a file to a specified URL using a multipart/form-data POST request.
    Args:
        url (str): The URL to which the file will be uploaded.
        path (str): The path to the file to be uploaded.
        file_name (str): The name of the file to be uploaded.
        meta_data (dict): Strict usage. The dict must have one key "metadata" and its value is a json dumped string
        entailing a dict with "name", "params" and "cv_acc".
    Returns:
        None: Prints upload success or error messages.
    """
    # Prepare files
    files = {'file': (file_name, open(os.path.join(path, file_name), 'rb'), 'file/joblib')}
    # Make the POST request
    response = requests.post(url, files=files, data=meta_data)

    if response.status_code == 200:
        print(f"File [{file_name}] upload successful.")
        print("Response content:", response.text)
    else:
        print(f"Error: {response.status_code} - {response.text}")


def main(args):
    """
    Main function for processing JSON data, training models, and saving outputs.

    Args:
        args (Any): Command-line arguments and options.

    Returns:
        None
    """

    # Get the data and process them into a Pandas data frame
    # json_data = get_json_data(args.url, args.query, args.aggregator)
    # data = process_json_data(json_data, args.aggregator)
    from data_receptor import main as data_receptor
    data = data_receptor(args)

    # Construct X and y for data matrix and target variable and create the KFold CV to ensure validation of models
    # across identical batches for exact and unbiased performance comparison
    X = data.drop('Label', axis=1)
    y = data['Label']
    output_dict = model_selection(X, y)

    # Make output path and save the per-class best models
    local_out_path = os.path.join(args.out_path, args.aggregator)
    if not os.path.exists(local_out_path):
        os.makedirs(local_out_path)

    for model in output_dict.keys():
        
        # Save the model file locally
        dump(output_dict[model]["model"], os.path.join(local_out_path, model + '.joblib'))
        
        # Save the metadata locally
        meta_data = output_dict[model]["metadata"]
        meta_data.to_csv(os.path.join(local_out_path, model + '.csv'))

        # Send models and meta-data to the API
        api_url_out = os.path.join(args.api_url, args.aggregator, "files")
        send_data(url=api_url_out,
                  path=local_out_path,
                  file_name=model + '.joblib',
                  meta_data={"metadata": json.dumps(meta_data.to_dict())})




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate linear models with k-fold cross-validation.")
    parser.add_argument("--aggregator", type=str, default='pfcpflowmeter', help="Aggregator to use. Can be 'pfcpflowmeter', 'tstat', 'cicflowmeter'. ")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--metric", type=str, default='f1_score', help="Metric to choose the best model from. Can be 'accuracy', 'f1_score', 'roc_auc', 'TPR', 'FPR'. ")
    parser.add_argument("--url", type=str, default="http://83.212.72.97:9200", help="API URL to request data.")
    parser.add_argument("--api_url", type=str, default='http://83.212.72.97:5000/analytics/', help="API URL to send models and metadata.")
    parser.add_argument("--out_path", type=str, default="./local_models", help="Where to store the models that will be send to the ElasticSearch storage service.")
    parser.add_argument("--query", type=dict, default={"_source": ["*"]}, help="Query to retrieve from ElasticSearch")

    args = parser.parse_args()
    main(args)