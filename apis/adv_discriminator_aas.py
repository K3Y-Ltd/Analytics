import os.path
from flask import Flask, request, jsonify, send_file
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
import os
import pandas as pd
from joblib import load
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack



app = Flask(__name__)
@app.route('/analytics/discriminator', methods=['POST'])
def api_call():
    try:
        # Extract parameters from JSON data
        aggregator = request.form.get('aggregator', 'pfcpflowmeter')
        model = str(request.form.get('model',  'LDA'))
        print("Body Parameters Received")

        print(request.files)

        if 'data_file' not in request.files:
            return jsonify({"status": "error", "message": "No data found in the request"}), 400

        if 'test_file' not in request.files:
            return jsonify({"status": "error", "message": "No test data found in the request"}), 400

        print("Data Received")

        data_file = request.files['data_file']
        if data_file.filename == '':
            return jsonify({"status": "error", "message": "No selected data file"}), 400

        test_file = request.files['test_file']
        if data_file.filename == '':
            return jsonify({"status": "error", "message": "No selected test file"}), 400

        data = pd.read_csv(data_file)
        test_data = pd.read_csv(test_file)

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

        test_data = test_data.drop(cols_dropped, axis=1)
        test_data = test_data.sort_index(axis=1)

        X_real = data.drop('Label', axis=1)

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

        X_adv = zoo_attack.generate(X_real.to_numpy())
        X_adv = pd.DataFrame(X_adv, columns=X_real.columns)

        X_real['Label'] = 0
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
        xgb_model.fit(X, y)
        preds = xgb_model.predict(test_data.drop('Label', axis=1))
        test_data.drop('Label', axis=1)["Label"] = preds
        output = BytesIO()
        test_data.to_csv(output, index=False)
        output.seek(0)
        print("done")
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='adversarial_predicted.csv')

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5004)

"""
curl -X POST http://127.0.0.1:5004/analytics/discriminator -F 'data_file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv' -F 'test_file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv' -o adversarial_predicted.csv
"""