# Analytics Inference

### Introduction
This script is designed to consume data from a Kafka topic, preprocess the data, make predictions using a pre-trained machine learning model, and then publish the results back to another Kafka topic. 
The model is downloaded from a remote API before starting the Kafka consumer. Depending on the type of data (e.g., from pfcpflowmeter, tstat, or cicflowmeter), specific columns are dropped during preprocessing to match the model's requirements.
The  Kafka Consumer subscribes to a topic and consumes messages (data instances). Then the script uses a pre-trained model to make predictions on the incoming data
and publishes the prediction results to another Kafka topic. The output is the following:

-- Prediction Results: Sent back to the Kafka producer topic in JSON format.

### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py -e .``
7. CD into scripts: ``cd scripts/``

### Usage
To utilize the script, please run the following:
```python inference.py```

Parameters:\
-- aggregator: The type of data aggregator (e.g., pfcpflowmeter, tstat, cicflowmeter). \
-- folds: The number of folds for cross-validation. \
-- metric: The evaluation metric to select the best model (accuracy, f1_score, roc_auc, TPR, FPR). \
-- file: The CSV file containing the dataset. \
-- api_url: API URL to send models and metadata. \
-- out_path: Path to store the locally saved models and metadata. \
-- query: Query to retrieve data from ElasticSearch, if not local. \
-- url	ElasticSearch API to request data from, if not local. \

### Example:
python inference.py \
    --url http://83.212.72.97:5000/analytics/ \
    --save_path ./downloaded_models \
    --model SVM.joblib \
    --broker_ip_address 83.212.72.97 \
    --broker_port 9094 \
    --group_id development \
    --consumer_topics pfcpflowmeter \
    --producer_topics pfcpflowmeter-inference

Please bear in mind that the script deploys the trained model and is executable only when storages services
are deployed too.
