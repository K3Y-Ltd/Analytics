# Analytics Adversarial Data Generator

### Introduction

This Flask application provides a service that generates adversarial examples using the ZOO (Zeroth Order Optimization) attack. The service accepts CSV files, processes the data based on the specified aggregator type, and generates adversarial examples using a pre-trained machine learning model.
evaluation metric to select the best model (accuracy, f1_score, roc_auc, TPR, FPR)

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
```python apis/adv_generator_aas.py```
The API will be available at http://127.0.0.1:5003. It enables the method
POST /analytics/generator. This endpoint allows users to upload a dataset and specify parameters (model for instance),
for generating an adversarial version of the uploaded dataset.
to use
Parameters:\
-- aggregator: The type of data aggregator (e.g., pfcpflowmeter, tstat, cicflowmeter). \
-- file: The CSV file containing the dataset. \
-- model: The model for which to generate the adversarial dataset i.e. the model to fool. \


### Example:
```bash
curl -X POST http://127.0.0.1:5003/analytics/generator
  -F 'aggregator=pfcpflowmeter'
  -F 'file=@/path/to/your/data.csv'
  -F 'model=LDA'
  -o adversarial.csv --max-time 10000000
```
