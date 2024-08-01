# Analytics Verificator

### Introduction

This Flask application provides an API endpoint for evaluating the robustness of a machine learning model against adversarial attacks. The service loads a pre-trained model, generates adversarial samples using the ZOO attack, and measures the performance degradation of the model when predicting on real versus adversarial data.
### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py sdist bdist_wheel``

### Usage
To utilize the API with some data, please run the following:
```python apis/verificator_aas.py```
The API will be available at http://127.0.0.1:5005. It enables the method
POST /analytics/verificator. This endpoint allows users to upload a dataset and specify parameters (model for instance),
for generating an adversarial version of the uploaded dataset.

Parameters:\

-- aggregator: The type of data aggregator. Options are pfcpflowmeter, tstat, and cicflowmeter. Default is pfcpflowmeter.\
-- model: The machine learning model for which to measure performance degradation. Default is LDA.\
-- file: The CSV file containing the data.

### Example:
```bash
curl -X POST http://127.0.0.1:5005/analytics/verificator \
     -F 'aggregator=pfcpflowmeter' \
     -F 'file=@/path/to/data.csv'
```
