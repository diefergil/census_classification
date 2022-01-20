import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from starter.config import DATA_DIR, CAT_FEATURES

import pytest
import pandas as pd

from starter.ml.data import process_data

@pytest.fixture(scope='session')
def data():
    """
    Get the training data
    """
    df = pd.read_csv(DATA_DIR / "census_cleaned.csv")
    
    return df

@pytest.fixture(scope='session')
def cat_features():
    return CAT_FEATURES

@pytest.fixture(scope='session')
def processed_output(data, cat_features):
    
    data_features, target, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
    )
    
    return {
        "input_data": data_features,
        "target": target,
        "encoder": encoder,
        "lb": lb
    }
    
@pytest.fixture(scope='session')
def positive_example():
    return {
        "age":47,
        "workclass":"Self-emp-inc",
        "fnlgt":181130,
        "education":"Prof-school",
        "education-num":15,
        "marital-status":"Married-civ-spouse",
        "occupation":"Prof-specialty",
        "relationship":"Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain":99999,
        "capital-loss":0,
        "hours-per-week":50,
        "native-country":"United-States"
        }
    
@pytest.fixture(scope='session')
def negative_example():
    return {
        "age":20,
        "workclass":"?",
        "fnlgt":133515,
        "education":"Some-college",
        "education-num":10,
        "marital-status":"Never-married",
        "occupation":"?",
        "relationship":"Own-child",
        "race":"White",
        "sex":"Female",
        "capital-gain":0,
        "capital-loss":0,
        "hours-per-week":15,
        "native-country":"France"
    }