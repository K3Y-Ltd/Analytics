# Analytics Data Receptor

### Introduction
This script is designed to retrieve, process, and clean data from different data aggregators such as pfcpflowmeter, tstat, and cicflowmeter. The script allows users to retrieve data from an ElasticSearch API, drop unnecessary columns based on the aggregator, and process the data into a format suitable for machine learning and other analytical purposes.
The script supports data retrieval from a remote API or local CSV files and outputs cleaned and structured data.
It outputs the:
-- Processed data: Cleaned and processed data for the given aggregator.



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
```python data_receptor.py```\


Parameters:\
--aggregator: (required) The type of data aggregator to process. Available options: pfcpflowmeter, tstat , cicflowmeter \
--url: (optional) The API URL to fetch data from, if Storage Services are up. Default: http://83.212.72.97:9200. \
--query: (optional) Query to retrieve data, if Storage Services are up. Default: { "_source": ["*"] }.


### Example:
python data_receptor.py --aggregator pfcpflowmeter








