import os
from flask import Flask, request, send_file, jsonify
from io import BytesIO
import pandas as pd
from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
from utils.utils import download_file
from joblib import load


def run_inference(file, aggregator, model_path, model_name):

    # Load the model and instance
    # url = 'http://83.212.74.114:5000/analytics/'
    # loaded_model = download_file(url, model_path, model_name)
    print(os.path.join(model_path, model_name))
    loaded_model = load(os.path.join(model_path, aggregator, model_name))
    print(f"Best parameters:  {loaded_model}")
    instance = pd.read_csv(file)
    print(instance)

    if aggregator == "pfcpflowmeter":
        cols_dropped = COLS_DROPPED_PFCP
    elif aggregator == "tstat":
        cols_dropped = COLS_DROPPED_TSTAT
    elif aggregator == "cicflowmeter":
        cols_dropped = COLS_DROPPED_CIC
    else:
        raise ValueError("Invalid aggregator type.")

    # Process the instance
    instance = instance.drop(cols_dropped, axis=1)
    instance = instance.drop('Label', axis=1)
    instance = instance.sort_index(axis=1)

    # Make a prediction using the machine learning model
    prediction = loaded_model.predict(instance)
    instance["Prediction"] = prediction
    return instance

app = Flask(__name__)
@app.route('/analytics/inference', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    aggregator = request.form.get('aggregator', 'pfcpflowmeter')
    model_path = request.form.get('model_path', 'downloaded_models')
    model_name = request.form.get('model_name', 'SVM.joblib')

    try:
        instance_with_prediction = run_inference(file, aggregator, model_path, model_name)
        print(instance_with_prediction)
        output = BytesIO()
        instance_with_prediction.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predicted_final.csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(port=5001)


"""
curl -X POST http://127.0.0.1:5001/analytics/inference \
  -F "file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv" \
  -F "aggregator=pfcpflowmeter" \
  -o predicted.csv
  # -F "url=http://83.212.72.97:5000/analytics/" \
  # -F "model_path=downloaded_models" \
  # -F "model_name=SVM.joblib" \
  """