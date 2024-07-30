from utils.datasets import COLS_DROPPED_PFCP, COLS_DROPPED_CIC, COLS_DROPPED_TSTAT
from io import BytesIO
import pandas as pd
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

@app.route('/analytics/data', methods=['POST'])
def api_call():
    try:
        # Extract parameters from JSON data
        aggregator = request.form.get('aggregator')

        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part in the request"}), 400

        print(request.files)
        file = request.files['file']
        print(file)
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
        print(data)

        output = BytesIO()
        data.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='processed.csv')

    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)


"""
curl -X POST http://127.0.0.1:5002/analytics/data -F 'aggregator=pfcpflowmeter' -F 'file=@/home/efklidis/ACROSS/data/training/pfcpflowmeter.csv'  -o processed.csv

"""