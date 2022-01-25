# Census classification with CI/CD

![PythonVersion](https://img.shields.io/badge/python-3.8%20|%203.9-success)
![example workflow](https://github.com//diefergil/census_classification/actions/workflows/python-app.yml/badge.svg)


## Overview

* Develop a classification model using [Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income) and CI/CD.
* The package was created following the course [Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)
* The artifactcts was registered using [`DVC`](https://dvc.org/) and `Amazon S3` as remote repository.
* The continous integration was made using [Github Actions](https://github.com/features/actions)
* The Model API was developed using [FASTApi](https://fastapi.tiangolo.com/)
* The modek is deployed using with Continuous Delivery on [Heroku](https://devcenter.heroku.com/)

## Train Model
   
   1. Clone the repository.
   2. Install requirements via `Conda` or `Pip`
   3. Put the census data in `/data` folder.
   4. Execute `python starter/train_model.py`
   5. The execution should have created 3 files: `model.pkl`, `encoder.pkl` and `label_binarizer.pkl`
   6. Check tests using `pytest -vv`

## API

POST requests are used to send data to the API.
You can use the API to predict the salary by:
- using the docs page on the Heroku at 
[API documentation](https://census-classification.herokuapp.com/docs#/default/predict_predict__post)
- use the requests for an individual using python request module (see [example in this repository](./call_api.py))
- use curl: an example curl command would be:

```bash
curl -X 'POST' \
  'https://census-classification.herokuapp.com/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 68,
  "workclass": "State-gov",
  "fnlgt": 146645,
  "education": "Doctorate",
  "education-num": 16,
  "marital-status": "Never-married",
  "occupation": "Farming-fishing",
  "relationship": "Unmarried",
  "race": "White",
  "sex": "Female",
  "capital-gain": 20051,
  "capital-loss": 0,
  "hours-per-week": 50,
  "native-country": "United-States"
}'
```
