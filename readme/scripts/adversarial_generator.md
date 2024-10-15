# Analytics Adversarial Generator

### Introduction

This script is designed to generate adversarial examples for machine learning models using the ZooAttack evasion method. The adversarial examples are generated based on pre-trained models (Logistic Regression, SVM, etc.) and specific data aggregators such as pfcpflowmeter, tstat, and cicflowmeter. The script also computes affinity metrics (correlation, MAE, Wasserstein distance) to evaluate the differences between the original and adversarial data.

The outputs of the script include:\
-- Adversarial datasets: saved in CSV format \
-- Metadata: containing attack parameters and affinity metrics.



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
```python adversarial_generator.py```\

Parameters:\
--aggregator: (required) The type of data aggregator to process. Available options: pfcpflowmeter, tstat, cicflowmeter \
--model: (required) The pre-trained model to use for generating adversarial examples. Available options: LR (Logistic Regression), SVM, etc.\
--attack: (required) The type of adversarial attack to perform. Currently, only ZOO (Zero Order Optimization) is supported.\
--query: (optional) Query to retrieve data from ElasticSearch. Default: {"_source": ["*"]}. \
--url: (optional) The API URL to download models from. Default: http://83.212.72.97:5000/analytics/. \
--save_path: (optional) The local path to save downloaded models. Default: ./downloaded_models. 

### Example:

python adversarial_generator.py --aggregator pfcpflowmeter --model LR --attack ZOO








