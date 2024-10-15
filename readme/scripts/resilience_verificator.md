# Analytics Resilience Verificator

### Introduction
This script is designed to evaluate the performance degradation of a machine learning model when subjected to adversarial attacks. It compares the model's performance on clean (actual) data and adversarially generated data and calculates the accuracy degradation caused by the attack. The adversarial datasets are pre-generated, and the script processes them to compute and store the results.

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
```python resilience_verificator.py```\

Parameters: \
--aggregator: The aggregator to use (e.g., tstat, pfcpflowmeter, cicflowmeter). Default is tstat.\
--query: A dictionary query to retrieve data from ElasticSearch (default: {"_source": ["*"]}).\
--model: The name of the trained model (e.g., LR, SVM). Default is LR.\
--attack: The type of adversarial attack (e.g., ZOO). Currently, only the ZOO attack is supported. Default is ZOO.\
--url: The URL of the API used for downloading models and metadata. Default is 'http://83.212.72.97:5000/analytics/'. \
--save_path: The directory path where the models will be saved after downloading. Default is ./downloaded_models.

### Example
python resilience_verificator.py --aggregator tstat --model LR --attack ZOO

