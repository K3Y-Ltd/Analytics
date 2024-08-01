# Analytics Adversarial Data Discriminator

### Introduction
This Flask application provides an API endpoint for generating adversarial data and training a discriminator model. The discriminator model is designed to differentiate between real and adversarial data samples. The application uses a combination of data preprocessing, adversarial attack generation, and machine learning to achieve this.

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
```python apis/adv_discriminator_aas.py```
The API will be available at http://127.0.0.1:5004. It enables the method
POST /analytics/discriminator. This endpoint allows users to train an XGBoost model in order
to differentiate in between real and adversarial samples

Parameters:\
-- aggregator: The type of data aggregator. Options are pfcpflowmeter, tstat, and cicflowmeter. Default is pfcpflowmeter.\
-- model: The machine learning model to use for adversarial attacks. Default is LDA.\
-- data_file: The CSV file containing the real data from which ZOO generates the adversarial data.\
-- test_file: The CSV file containing the test data.

### Example:
```bash
curl -X POST http://127.0.0.1:5003/analytics/generator
  -F 'aggregator=pfcpflowmeter'
  -F 'file=@/path/to/your/data.csv'
  -F 'model=LDA'
  -o adversarial.csv --max-time 10000000
```
