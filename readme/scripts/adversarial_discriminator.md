# Analytics Adversarial Data Discriminator

### Introduction
This script is designed to evaluate the performance of adversarial examples generated for different machine learning models using the XGBoost classifier. The script compares the adversarial data with real data to assess the effectiveness of adversarial attacks. It trains and tests an XGBoost model to classify adversarial and real data, calculating metrics such as accuracy and F1 score through cross-validation and test evaluation.
### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``python setup.py -e .``
6. Install the package: ``python setup.py sdist bdist_wheel``
7. CD into scripts: ``cd scripts/``

### Usage
To utilize the script, please run the following:
```python adversarial_discriminator.py```\

Parameters: \
--aggregator: (required) The data aggregator to use. Available options: pfcpflowmeter, tstat, cicflowmeter \
--query: (optional) Query to retrieve data from ElasticSearch. Default: {"_source": ["*"]}. \
--model: (required) The pre-trained model to evaluate against adversarial examples (e.g., LR for Logistic Regression, SVM for Support Vector Machine). \
--attack: (required) The adversarial attack method used for generating adversarial examples. Currently, only ZOO (Zero Order Optimization) is supported. \
--url: (optional) The API URL for downloading models. Default: http://83.212.72.97:5000/analytics/. \
--save_path: (optional) The local path to save downloaded models. Default: ./downloaded_models.

### Example
python adversarial_discriminator.py --aggregator tstat --model LR --attack ZOO
