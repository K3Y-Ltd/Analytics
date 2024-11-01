import os
import argparse
import pandas as pd
from data_receptor import main as data_receptor
from joblib import load
from utils.utils import accuracy_score, f1_score
import glob

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

def main(args):
    # Load the original, actual data
    data = data_receptor(args)
    X = data.drop('Label', axis=1)
    y = data['Label']

    # Load the trained model for the given class
    modelname = args.model + '.joblib'
    model = load(os.path.join('../local_models', args.aggregator, modelname))

    # Load the model once the functionality is back from Vagos
    # url = os.path.join(args.url, args.aggregator)
    # save_path = os.path.join(args.save_path, args.aggregator)
    # model = download_file(url, save_path, modelname)

    # Load all attacks generated by the adversarial generator for the specific model and attack class
    path = os.path.join("../data/adversarial", args.aggregator, args.model)
    attack_datasets = [x for x in glob.glob(os.path.join(path, "*.csv")) if "metadata" not in x]

    # Compute performance before the attack
    metrics = model_evaluate(model, X, y)
    print(f"Performance on the actual data computed: {str(metrics)}")
    df = pd.DataFrame(columns=['before', "after", "degradation"])

    for idx, attack_dataset in enumerate(attack_datasets, 0):

        X_adv = pd.read_csv(attack_dataset).drop('Label', axis=1)
        metrics_adv = model_evaluate(model, X_adv, y)
        acc_degrade = 100 * (metrics[0] - metrics_adv[0]) / metrics[0]
        print(f"Performance on the adversarial data computed: {str(metrics_adv)}")

        # save metadata to the same folder for efficient retrieval
        df.loc[len(df)] = [str(metrics), str(metrics_adv), acc_degrade]
        df.to_csv(os.path.join("../data/adversarial", args.aggregator,
                               args.model, f"{args.attack}-metadata.csv"), index=False)


    print(f"Average accuracy degradation of {args.model} for the {args.attack} attack is {df['degradation'].mean()}")





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