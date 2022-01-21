import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
import pytest
from constants import data, cat_features, processed_output

def test_categorical_columns(data, cat_features):
    """
    Check split have same number of rows for X and y
    """
    cats = list(data.select_dtypes(include='object').columns)[:-1]

    assert cats == cat_features

def test_label_binarizer(processed_output):
    assert len(processed_output['lb'].classes_) == 2
    
def test_encoder(processed_output, cat_features):
    assert processed_output['encoder'].n_features_in_ == len(cat_features)