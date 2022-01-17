import os
import sys
import numpy as np
import pandas as pd
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pytest
from starter.config import DATA_DIR

from starter.ml import model as ml_model
from constants import data, cat_features, processed_output

@pytest.fixture(scope='session')
def model(processed_output):
    model = ml_model.train_model(
        processed_output["input_data"], processed_output["target"], 
        model=RandomForestClassifier()
        )
    
    return model

def test_train_model(model, processed_output):
    preds = ml_model.inference(model, processed_output["input_data"])
    assert hasattr(model, "predict")
    
def test_inference(model, processed_output):
    preds = ml_model.inference(model, processed_output["input_data"])
    assert isinstance(preds, np.ndarray)
    
    
def test_compute_model_metrics():
    y = np.array([0, 0, 0 ,1 , 1, 1])
    y_pred = np.array([0, 0, 0 ,1 , 1, 1])
    
    metrics_dict = ml_model.compute_model_metrics(y, y_pred)
    assert isinstance(metrics_dict, dict)