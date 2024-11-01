# Analytics Training

### Introduction

This Flask application provides a data processing service that accepts CSV files, processes the data based on the specified aggregator type, and returns the processed data as a CSV file. The service drops specific columns from the input data according to the aggregator type and sorts the columns.

### Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``python setup.py -e .``
6. Install the package: ``python setup.py sdist bdist_wheel``

### Usage
To utilize the API with some data, please run the following:
```python apis/data_receptor_aas.py```
The API will be available at http://127.0.0.1:5002. It enables the method
POST /analytics/data. This endpoint allows users to upload a dataset and receive back its processed version.
You can process data by sending a POST request to the /analytics/data endpoint with the following parameters:

Parameters:\
-- file: The CSV file containing the data to be processed.\
-- aggregator: The type of aggregator (e.g., pfcpflowmeter, tstat, cicflowmeter).

### Example:
```bash
curl -X POST http://127.0.0.1:5002/analytics/data \
  -F 'aggregator=pfcpflowmeter' \
  -F 'file=@/path/to/your/data.csv' \
  -o processed.csv
```



