import os
import argparse
import pandas as pd
from data_receptor import main as data_receptor
from utils.utils import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def model_evaluate(model, X, y):

    # Predict on the test set and evaluate
    y_probs = model.predict_proba(X) if model.__class__.__name__ not in ['LinearSVC'] else model._predict_proba_lr(X)

    # Make the class mapping and evaluate all possible metrics
    class_mapping = {label:label_idx  for label_idx, label in enumerate(model.classes_)}
    metrics = [metric(y, y_probs, class_mapping)
               for metric in [accuracy_score, f1_score]]
    return metrics
acc = []
def main(args):
    # Load the original, actual data

    for i in range(16):
        data = data_receptor(args)
        X_real = data.drop('Label', axis=1)
        X_real['Label'] = 0

        data_adv = pd.read_csv(os.path.join('../data/adversarial', args.aggregator, args.model, args.attack + f"-{i}.csv"))
        X_adv = data_adv.drop('Label', axis=1)
        X_adv['Label'] = 1

        dataset = pd.concat([X_real, X_adv])
        X = dataset.drop('Label', axis=1)
        y = dataset['Label']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize XGBoost classifier
        xgb_model = XGBClassifier()

        # Perform cross-validation on training data
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)

        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())

        # Train the model on the full training data
        xgb_model.fit(X_train, y_train)

        # Evaluate the model on the test set
        test_accuracy = xgb_model.score(X_test, y_test)
        print("Test set accuracy:", test_accuracy)
        acc.append(test_accuracy)
    print(acc)
    print(sum(acc)/len(acc))





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive data required for adversarial data generation.")
    parser.add_argument("--aggregator", type=str, default='tstat', help="Aggregator to use. Can be 'pfcpflowmeter', 'tstat', 'cicflowmeter'. ")
    parser.add_argument("--query", type=dict, default={"_source": ["*"]}, help="Query to retrieve from ElasticSearch")
    parser.add_argument("--model", type=str, default="LR", help="Trained AI model to choose from.")
    parser.add_argument("--attack", type=str, default="ZOO", help="Adversarial attack to choose from. Currently only ZOO is supported.")
    parser.add_argument("--url", type=str, default='http://83.212.72.97:5000/analytics/', help="API URL to send models and metadata.")
    parser.add_argument("--save_path", type=str, default="./downloaded_models", help="Where to store the models downloaded from the ElasticSearch storage service.")
    # parser.add_argument("--model", type=str, default="SVM.joblib", help="Name for the model(s) downloaded from the ElasticSearch storage service.")

    args = parser.parse_args()
    main(args)