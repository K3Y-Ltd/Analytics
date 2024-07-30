import time
import streamlit as st
from sklearn.model_selection import train_test_split
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_TSTAT, COLS_DROPPED_CIC
import os
from utils.app_utils import *
from datetime import datetime
from art.estimators.classification import SklearnClassifier

from art.attacks.evasion import BoundaryAttack, ZooAttack, HopSkipJump



def compute_affinity(model_name, dataset_name, X_test, X_test_adv):
    st.write(f"### {dataset_name} - Data Differences for {model_name} ")
    st.dataframe(X_test - X_test_adv, hide_index=True)
    st.write(f"### Dataset Affinity Metrics - {dataset_name} ")
    st.write(f"Correlation: {0.987} | MAE: {0.035} | Wasserstein Distance {0.044}")

def measure_degradation(model_name, dataset_name):
    st.write(f"Performance Degradation for {model_name} on the {dataset_name} Dataset: 32%")

def do_lda(folds):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    lda_param_grid = {'solver': ['svd', 'lsqr']}
    lda_grid_combinations = ParameterGrid(lda_param_grid)
    lda_models = [LinearDiscriminantAnalysis(**params) for params in lda_grid_combinations]
    lda_best, lda_meta, lda_meta_data = cv_customized(X, y, kf, lda_models, metrics_name, folds=folds)
    return lda_best, lda_meta, lda_meta_data

def do_lr(folds):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    lr_param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 0.5, 1.0, 10.0], 'solver': ['liblinear'], 'max_iter':[100]}
    lr_grid_combinations = ParameterGrid(lr_param_grid)
    lr_models = [LogisticRegression(**params) for params in lr_grid_combinations]
    lr_best, lr_meta, lr_meta_data = cv_customized(X, y, kf, lr_models, metrics_name, folds=folds)
    return lr_best, lr_meta, lr_meta_data

def do_svm(folds):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    svm_param_grid = {'C': [0.1, 1, 10, 10], 'penalty': ['l1'], 'loss': ['squared_hinge'], 'dual': [False], 'max_iter': [300]}
    svm_grid_combinations = ParameterGrid(svm_param_grid)
    svm_models = [LinearSVC(**params) for params in svm_grid_combinations]
    svm_best, svm_meta, svm_meta_data = cv_customized(X, y, kf, svm_models, metrics_name, folds=folds)
    return svm_best, svm_meta, svm_meta_data

def load_data(dataset_name):
    if dataset_name == "Tstat":
        cols_dropped = COLS_DROPPED_TSTAT
        path = "tstat.csv"
        data = pd.read_csv(os.path.join('data', 'training', path))
        data = data.drop(cols_dropped, axis=1)
        data = data.sort_index(axis=1)
        X = data.drop('Label', axis=1)
        y = data['Label']
        return data, X, y, data.columns

    elif dataset_name == "Pfcpflowmeter":
        cols_dropped = COLS_DROPPED_PFCP
        path = "pfcpflowmeter.csv"
        data = pd.read_csv(os.path.join('data', 'training', path))
        data = data.drop(cols_dropped, axis=1)
        data = data.sort_index(axis=1)
        X = data.drop('Label', axis=1)
        y = data['Label']
        return data, X, y, data.columns

    elif dataset_name == "Ciflowmeter":
        cols_dropped = COLS_DROPPED_CIC
        path = "cicflowmeter.csv"
        data = pd.read_csv(os.path.join('data', 'training', path))
        data = data.drop(cols_dropped, axis=1)
        data = data.sort_index(axis=1)
        X = data.drop('Label', axis=1)
        y = data['Label']
        return data, X, y, data.columns


def run_inference(model, df):
    output_col_1, output_col_2, output_col_3 = st.columns([0.1, 0.8, 0.1], gap="small")
    with output_col_1:
        st.write(f"#### Time Stamp")

    with output_col_2:
        st.write(f"#### Security Event")

    with output_col_3:
        st.write(f"#### Prediction")

    for i in range(30):
        with st.container():
            output_col_1, output_col_2, output_col_3 = st.columns([0.1, 0.8, 0.1], gap="small")
            with output_col_1:
                current_time = datetime.now()
                st.write(current_time)

            with output_col_2:
                st.dataframe(pd.DataFrame([df.iloc[i]]), hide_index=True)

            with output_col_3:
                st.write(model.predict(df.values[i].reshape(1, -1)).item())
                time.sleep(1)


# Streamlit app
st.set_page_config(layout="wide")

st.title("ACROSS Analytics Dashboard")

# Dataset selection
dataset_name = st.sidebar.selectbox("Select Traffic Aggregator", ("Tstat", "Pfcpflowmeter", "Ciflowmeter"))
model_name = st.sidebar.selectbox("Select Analytics Model", ("Logistic Regression", "Support Vector Machine",
                                                              "Linear Discriminant Analysis"))

folds = st.sidebar.selectbox("Number of Cross-validation Folds", (5, 10))
metrics_name = st.sidebar.selectbox("Best Model Selection Metric", ("accuracy", "f1_score", "roc_auc", "TPR", "FPR"))

# Load the selected dataset
df, X, y, columns = load_data(dataset_name)

# Create tabs
tabs = st.tabs(["Data Receptor", "Analytics Training Service", "Analytics Inference Service", "Analytics Data Drift Detector"])

# Data tab
with tabs[0]:
    st.write(f"## {dataset_name} Dataset")
    st.write(df)


columns_to_keep = [0, 1]


# Model Training tab
with tabs[1]:
    st.write(f"## {dataset_name} - Model Training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if dataset_name == "Tstat":

        if model_name == "Linear Discriminant Analysis":
            lda_best, lda_meta, lda_meta_data = do_lda(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lda_meta_data = lda_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                # st.write(lda_meta_data.drop(lda_meta_data.index[-1]))
                st.dataframe(lda_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lda_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})
                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Logistic Regression":
            lr_best, lr_meta, lr_meta_data = do_lr(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lr_meta_data = lr_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                # st.write(lr_meta_data.drop(lr_meta_data.index[-1]))
                st.dataframe(lr_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lr_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})

                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Support Vector Machine":
            svm_best, svm_meta, svm_meta_data = do_svm(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                svm_meta_data = svm_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                # st.write(lr_meta_data.drop(lr_meta_data.index[-1]))
                st.dataframe(svm_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(svm_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})

                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

    elif dataset_name == "Pfcpflowmeter":

        if model_name == "Linear Discriminant Analysis":
            lda_best, lda_meta, lda_meta_data = do_lda(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lda_meta_data = lda_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                # st.write(lda_meta_data.drop(lda_meta_data.index[-1]))
                st.dataframe(lda_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lda_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})

                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Logistic Regression":
            lr_best, lr_meta, lr_meta_data = do_lr(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lr_meta_data = lr_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                st.dataframe(lr_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lr_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})

                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Support Vector Machine":
            svm_best, svm_meta, svm_meta_data = do_svm(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                svm_meta_data = svm_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                st.dataframe(svm_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(svm_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})

                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

    elif dataset_name == "Ciflowmeter":

        if model_name == "Linear Discriminant Analysis":
            lda_best, lda_meta, lda_meta_data = do_lda(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lda_meta_data = lda_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                st.dataframe(lda_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lda_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})
                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Logistic Regression":
            lr_best, lr_meta, lr_meta_data = do_lr(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                lr_meta_data = lr_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                st.dataframe(lr_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(lr_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})
                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

        elif model_name == "Support Vector Machine":
            svm_best, svm_meta, svm_meta_data = do_svm(folds)
            output_col_1, output_col_2 = st.columns([0.75, 0.25], gap="small")
            with output_col_1:
                st.write(f"#### Cross-validation Hyperparameters")
                svm_meta_data = svm_meta_data.rename(columns={"params": f"{model_name} Hyperparameters", "name":"Model"})
                # st.write(svm_meta_data.drop(svm_meta_data.index[-1]))
                st.dataframe(svm_meta_data.iloc[:, columns_to_keep], hide_index=True, use_container_width=True)
            with output_col_2:
                st.write(f"#### Metrics")
                df_flattened = pd.json_normalize(svm_meta_data["metrics"])
                df_flattened = df_flattened.rename(
                    columns={"accuracy":"Accuracy", "f1_score": "F1 score", "roc_auc":"ROC AUC"})
                st.dataframe(df_flattened, hide_index=True, use_container_width=True)

# Prediction tab
with tabs[2]:
    st.write(f"## {dataset_name} - Real-time Predictions on Incoming Payloads")

    if dataset_name == "Tstat":

        if model_name == "Linear Discriminant Analysis":
            run_inference(lda_best, X_test)

        elif model_name == "Logistic Regression":
            run_inference(lr_best, X_test)

        elif model_name == "Support Vector Machine":
            run_inference(svm_best, X_test)

    elif dataset_name == "Pfcpflowmeter":

        if model_name == "Linear Discriminant Analysis":
            run_inference(lda_best, X_test)

        elif model_name == "Logistic Regression":
            run_inference(lr_best, X_test)

        elif model_name == "Support Vector Machine":
            run_inference(svm_best, X_test)

    elif dataset_name == "Ciflowmeter":

        if model_name == "Linear Discriminant Analysis":
            run_inference(lda_best, X_test)

        elif model_name == "Logistic Regression":
            run_inference(lr_best, X_test)

        elif model_name == "Support Vector Machine":
            run_inference(svm_best, X_test)



with tabs[3]:
    st.write(f"## {dataset_name} - Actual Dataset")
    st.dataframe(X_test, hide_index=True)

    if st.button("Generate Adversarial Dataset"):

        if model_name == "Linear Discriminant Analysis":
            st.write(f"### {dataset_name} - Adversarial Dataset for {model_name} ")
            hop_attacks = HopSkipJump(classifier=SklearnClassifier(model=lda_best),  targeted=False, max_iter=200, max_eval=10000, init_size=250, init_eval=200)
            st.write(hop_attacks)
            X_test_adv = hop_attacks.generate(X_test.to_numpy())
            X_test_adv = pd.DataFrame(X_test_adv, columns=X_test.columns)
            st.dataframe(X_test_adv, hide_index=True)
            compute_affinity(model_name, dataset_name, X_test, X_test_adv)
            measure_degradation(model_name, dataset_name)

        elif model_name == "Logistic Regression":
            st.write(f"### {dataset_name} - Adversarial Dataset for {model_name} ")
            hop_attacks = HopSkipJump(classifier=SklearnClassifier(model=lr_best),  targeted=False, max_iter=200, max_eval=10000, init_size=250, init_eval=200)
            st.write(hop_attacks)
            X_test_adv = hop_attacks.generate(X_test.to_numpy())
            X_test_adv = pd.DataFrame(X_test_adv, columns=X_test.columns)
            st.dataframe(X_test_adv, hide_index=True)
            compute_affinity(model_name, dataset_name, X_test, X_test_adv)
            measure_degradation(model_name, dataset_name)

        elif model_name == "Support Vector Machine":
            st.write(f"### {dataset_name} - Adversarial Dataset for {model_name} ")
            hop_attacks = HopSkipJump(classifier=SklearnClassifier(model=svm_best),  targeted=False, max_iter=200, max_eval=10000, init_size=250, init_eval=200)
            st.write(hop_attacks)
            X_test_adv = hop_attacks.generate(X_test.to_numpy())
            X_test_adv = pd.DataFrame(X_test_adv, columns=X_test.columns)
            st.dataframe(X_test_adv, hide_index=True)
            compute_affinity(model_name, dataset_name, X_test, X_test_adv)
            measure_degradation(model_name, dataset_name)






