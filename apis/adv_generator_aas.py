import os.path
from flask import Flask, request, jsonify, send_file
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
import numpy as np
from io import BytesIO

import os
import pandas as pd
from joblib import load
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

np.seterr(divide='ignore', invalid='ignore')
simplefilter("ignore", category=(ConvergenceWarning))



app = Flask(__name__)
@app.route('/analytics/generator', methods=['POST'])
def api_call():
    try:
        # Extract parameters from JSON data
        aggregator = request.form.get('aggregator', 'pfcpflowmeter')
        model = str(request.form.get('model',  'LDA'))
        print("Body Parameters Received")

        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part in the request"}), 400

        print("Data Received")
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        data = pd.read_csv(file)

        if aggregator == "pfcpflowmeter":
            cols_dropped = COLS_DROPPED_PFCP

        elif aggregator == "tstat":
            cols_dropped = COLS_DROPPED_TSTAT

        elif aggregator == "cicflowmeter":
            cols_dropped = COLS_DROPPED_CIC

        else:
            raise ValueError("Invalid aggregator type.")

        data = data.drop(cols_dropped, axis=1)
        data = data.sort_index(axis=1)

        X = data.drop('Label', axis=1)
        y = data['Label']

        print("Data Loaded")
        zoo_attack_params = {'binary_search_steps': 20,
                           'max_iter': 20, 'nb_parallel': 1,
                           'variable_h': 0.1,
                           'confidence': 0.1}
        modelname = model + '.joblib'
        model = load(os.path.join('local_models', aggregator, modelname))
        print("Model Loaded")

        # url = os.path.join(args.url, args.aggregator)
        # save_path = os.path.join(args.save_path, args.aggregator)
        # model = download_file(url, save_path, modelname)

        zoo_attack = ZooAttack(classifier=SklearnClassifier(model=model), **zoo_attack_params)
        print("Attacks Initialized")

        path = os.path.join("data/adversarial", aggregator, os.path.basename(modelname))
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Directory Created")

        X_adv = zoo_attack.generate(X.to_numpy())
        # save as data frame to designated folders
        X_adv = pd.DataFrame(X_adv, columns=X.columns)
        X_adv['Label'] = y
        X_adv_path = os.path.join("./data/adversarial", aggregator,
                                  os.path.basename(modelname), "ZOO.csv")
        X_adv.to_csv(X_adv_path, index=False)
        output = BytesIO()
        X_adv.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='adversarial.csv')

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5003)

"""
curl -X POST http://127.0.0.1:5003/analytics/generator -F 'aggregator=pfcpflowmeter' -F 'file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv' -o adversarial.csv --max-time 1000000000

"""