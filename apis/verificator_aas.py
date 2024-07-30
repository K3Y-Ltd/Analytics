import os
import pandas as pd

from joblib import load
from utils.utils import accuracy_score, f1_score
import os.path
from flask import Flask, request, jsonify
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
@app.route('/analytics/verificator', methods=['POST'])
def discriminator():
    # Load the original, actual data
    try:
        # Extract parameters from JSON data
        aggregator = request.form.get('aggregator', 'pfcpflowmeter')
        model = str(request.form.get('model',  'LDA'))
        print("Body Parameters Received.")
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No data found in the request"}), 400
        print("Data Received.")

        # Load the data from file
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
        data = pd.read_csv(file)
        print("Data Loaded.")

        # Process according to aggregator
        if aggregator == "pfcpflowmeter":
            cols_dropped = COLS_DROPPED_PFCP
        elif aggregator == "tstat":
            cols_dropped = COLS_DROPPED_TSTAT

        elif aggregator == "cicflowmeter":
            cols_dropped = COLS_DROPPED_CIC
        else:
            raise ValueError("Invalid aggregator type.")

        # Process real data and split
        data = data.drop(cols_dropped, axis=1)
        data = data.sort_index(axis=1)
        X_real = data.drop('Label', axis=1)
        y_real = data['Label']
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2)
        print("Data Processed.")

        # Initialize attack and model
        zoo_attack_params = {'binary_search_steps': 20,
                           'max_iter': 20, 'nb_parallel': 1,
                           'variable_h': 0.1,
                           'confidence': 0.1}
        modelname = model + '.joblib'
        model = load(os.path.join('local_models', aggregator, modelname))
        print("Model Loaded Locally.")

        # url = os.path.join(args.url, args.aggregator)
        # save_path = os.path.join(args.save_path, args.aggregator)
        # model = download_file(url, save_path, modelname)
        # print("Model Loaded Remotely.")

        zoo_attack = ZooAttack(classifier=SklearnClassifier(model=model), **zoo_attack_params)
        print("Attacks Initialized")


        X_test_adv = zoo_attack.generate(X_test_real.to_numpy())
        y_test_adv = y_test_real.copy()
        data_adv = pd.DataFrame(X_test_adv, columns=X_real.columns)
        data_adv['Label'] = y_test_adv
        print("Adversarial Data Generated.")

        # Measure performance
        preds_real = model.predict(X_test_real)
        preds_adv = model.predict(data_adv.drop("Label", axis=1))
        print("Performance Computed.")

        # Measure degradation
        accuracy_real = sum(preds_real == y_test_real) / len(preds_real)
        accuracy_adv = sum(preds_adv == y_test_adv) / len(preds_adv)
        print(f"Accuracy degraded from {accuracy_real} to {accuracy_adv}")
        return {"message": "Success"}, 200

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5005)

"""
curl -X POST http://127.0.0.1:5005/analytics/verificator -F 'aggregator=pfcpflowmeter' -F 'file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv'
"""