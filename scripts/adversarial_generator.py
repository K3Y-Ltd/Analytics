import os
import argparse
import pandas as pd
from data_receptor import main as data_receptor
from utils.utils import download_file, affinity
from joblib import load
from sklearn.model_selection import ParameterGrid
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, BoundaryAttack, HopSkipJump


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # for a specific model and aggregator

    data = data_receptor(args)
    X = data.drop('Label', axis=1)
    y = data['Label']

    zoo_attack_grid = {'binary_search_steps': [10, 20],
                       'max_iter': [10, 20], 'nb_parallel': [1],
                       'variable_h': [0.05, 0.2],
                       'confidence': [0.1, 0.05]}
    zoo_combinations = ParameterGrid(zoo_attack_grid)
    modelname = args.model + '.joblib'
    model = load(os.path.join('../local_models', args.aggregator, modelname))

    # url = os.path.join(args.url, args.aggregator)
    # save_path = os.path.join(args.save_path, args.aggregator)
    # model = download_file(url, save_path, modelname)

    zoo_attacks = [ZooAttack(classifier=SklearnClassifier(model=model), **params) for params in zoo_combinations]

    df_attacks = pd.DataFrame(columns=['attacks', "correlation", "mae", "wasserstein"])
    path = os.path.join("../data/adversarial", args.aggregator, args.model)
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, attack in enumerate(zoo_attacks, 0):

        # generate adversarial data and measure affinity
        X_adv = attack.generate(X.to_numpy())
        corr, mae, w = affinity(X.to_numpy(), X_adv)

        # save as data frame to designated folders
        X_adv = pd.DataFrame(X_adv, columns=X.columns)
        X_adv['Label'] = y
        X_adv.to_csv(os.path.join("../data/adversarial", args.aggregator,
                                  args.model, f"{args.attack}-{idx}.csv"), index=False)

        # save metadata to the same folder for efficient retrieval
        df_attacks.loc[len(df_attacks)] = [str(zoo_combinations[idx]), corr, mae, w]
        df_attacks.to_csv(os.path.join("../data/adversarial", args.aggregator,
                                       args.model, f"{args.attack}-metadata.csv"), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive data required for adversarial data generation.")
    parser.add_argument("--aggregator", type=str, default='pfcpflowmeter', help="Aggregator to use. Can be 'pfcpflowmeter', 'tstat', 'cicflowmeter'. ")
    parser.add_argument("--query", type=dict, default={"_source": ["*"]}, help="Query to retrieve from ElasticSearch")
    parser.add_argument("--model", type=str, default="LR", help="Trained AI model to choose from.")
    parser.add_argument("--attack", type=str, default="ZOO", help="Adversarial attack to choose from. Currently only ZOO is supported.")
    parser.add_argument("--url", type=str, default='http://83.212.72.97:5000/analytics/', help="API URL to send models and metadata.")
    parser.add_argument("--save_path", type=str, default="./downloaded_models", help="Where to store the models downloaded from the ElasticSearch storage service.")
    # parser.add_argument("--model", type=str, default="SVM.joblib", help="Name for the model(s) downloaded from the ElasticSearch storage service.")

    args = parser.parse_args()
    main(args)