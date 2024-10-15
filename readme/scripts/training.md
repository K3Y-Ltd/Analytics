# Analytics Training

### Introduction
This script is designed for training, cross-validating, and evaluating several linear machine learning models (Logistic Regression, Linear Discriminant Analysis, and Linear Support Vector Machines) using k-fold cross-validation. The script also supports sending trained models and their metadata to a remote API.
It performs k-fold cross-validation on a set of linear models. Thereafter, it automatically selects the best-performing model based on a specified evaluation metric.
It saves the best models and metadata locally, while supports sending trained models and their corresponding metadata to the Block Object & File Storage.
If the connection times out, they are only saved locally. Moreover, the script calculates several evaluation metrics, including accuracy, F1 score, ROC AUC, true positive rate (TPR), and false positive rate (FPR), and allws the user to save the model with the
best performance on one of the metrics above, specified as an argument. The script generates the following outputs:

-- Model Files: Saved in .joblib format. \
-- Metadata Files: Saved in .csv format, containing the evaluation metrics and parameters for the best model. \
-- API Response: If the remote upload is successful, the server response is printed. \

### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py sdist bdist_wheel``
7. CD into scripts: ``cd scripts/``

### Usage
To utilize the script, please run the following:
```python training.py```

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
python training.py --aggregator pfcpflowmeter --folds 5 --metric f1_score --url http://83.212.72.97:9200 --api_url http://83.212.72.97:5000/analytics/ --out_path ./local_models --query "{\"_source\": [\"*\"]}"

