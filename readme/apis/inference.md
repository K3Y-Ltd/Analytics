# Analytics Inference


### Introduction
This Flask application provides an inference service that uses pre-trained machine learning models to make predictions 
on incoming data. The service accepts CSV files, processes the data, and returns predictions. The models and data 
processing pipelines are specifically tailored for different types of data aggregators.


### Installation
Ensure you have Python 3 installed. Optionally make a virtual environment:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py -e .``


### Usage
To utilize the API with some data, please run the following: 
```python apis/inference_aas.py``` The API will be available at http://127.0.0.1:5001. It enables the method POST 
/analytics/inference. This endpoint allows users to upload a data sample or a batch of samples in CSV files,
processes them according to the specified aggregator type, loads aggregator-specific pre-trained models, makes predictions
and returns them as a CSV file.


You can make predictions by sending a POST request to the /analytics/inference endpoint with the following parameters:

-- file: The CSV file containing the data to be predicted.\
-- aggregator: The type of aggregator (e.g., pfcpflowmeter, tstat, cicflowmeter).\
-- model_path: The path to the directory containing the pre-trained models.\
-- model_name: The name of the pre-trained model file.


### Example
Execute the API call with the desired arguments:
```bash
curl -X POST http://127.0.0.1:5001/analytics/inference
  -F "file=@/path/to/your/data.csv"
  -F "aggregator=pfcpflowmeter"
  -o predicted.csv
```
