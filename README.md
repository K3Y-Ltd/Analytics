![alt text](logos/logo_across.jpg )

# ACROSS - Analytics & Data Drift Detector

This repository contains the Analytics and the Data Drift Detector services developed for the ACROSS project.

## Introduction
It contains several components:
1) Training (Analytics)
2) Inference (Analytics)
3) Data Receptor (Data Drift Detector)
4) Adversarial Generator (Data Drift Detector)
5) Adversarial Discriminator (Data Drift Detector)
6) Resilience Verificator (Data Drift Detector)

Each component is provided as an API with usage examples as well as in scripts, able to be run from the terminal.
##

## Installation
To install the service please follow the steps enumerated below:
1. Clone the repository: ``git clone https://github.com/K3Y-Ltd/Analytics.git``
2. CD into the working directory ``cd Analytics/``
3. Make virtual environment ```virtualenv -p python3.10 venv```
4. Activate the environment ``source venv/bin/activate``
5. Install the requirements: ``pip install -r requirements.txt``
6. Install the package: ``python setup.py -e .``

 For more information on each component please refer to the dedicated Readme files.


![alt text](logos/logo_k3y.png )
