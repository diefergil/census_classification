import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from starter.config import DATA_DIR

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
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

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