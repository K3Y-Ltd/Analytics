# Analytics Training

### Introduction

This README provides an overview of the project, instructions for installation, and usage examples for the API. It also 
outlines the main functions and their purposes within the code.
The primary functionality includes loading data, training multiple models, performing cross-validation, and selecting
the best model based on specified metrics. Trained models and their metadata are saved and optionally uploaded to a server.

### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py -e .``

### Usage
To utilize the API with some data, please run the following:
```python apis/training_aas.py```
The API will be available at http://127.0.0.1:5000. It enables the method
POST /analytics/training. This endpoint allows users to upload a dataset and specify parameters for model training and evaluation.

Parameters:\
-- aggregator: The type of data aggregator (e.g., pfcpflowmeter, tstat, cicflowmeter). \
-- folds: The number of folds for cross-validation. \
-- metric: The evaluation metric to select the best model (accuracy, f1_score, roc_auc, TPR, FPR). \
-- file: The CSV file containing the dataset.


### Example:
``curl -X POST http://127.0.0.1:5000/analytics/training
     -F 'aggregator=pfcpflowmeter'
     -F 'folds=5'
     -F 'metric=accuracy'
     -F 'file=@/path/to/your/data.csv'
     --output final.zip``



