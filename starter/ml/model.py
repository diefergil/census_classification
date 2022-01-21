from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
from typing import Dict
import logging

logger = logging.getLogger("root")
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, model, hyperparameters=None):
    """
    Trains a machine learning model and returns it.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model : An object with fit method.
    Returns
    -------
    model
        Trained machine learning model.
    """
    if hyperparameters:
        logger.info("Tuning model")
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=3, scoring='f1')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info("Best params in grid_search: %s", grid_search.best_params_)
    else:
        model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {
        'precision': precision, 
        'recall': recall, 
        'fbeta':fbeta
        }


def inference(model, X):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : A sklearn model fitted
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    
    return preds

def calculate_slice_performance(data: pd.DataFrame, group_name: str, true_column: str, predicted_column:str) -> Dict:
    """calculate_slice_performance Calculate classification 
    metrics based on slices

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with a object column,
        true target and predicted target
    group_name: str
        The name of the column group
    true_column: str
        column name of the true value
    predicted_column:
        column name of the predicted value

    Returns
    -------
    Dict
        A dict whit the performance in every slice
    """    
    performance_group = dict()
    
    data_grouped = data.groupby(group_name)
    for group in data_grouped:
        slice_name = group[0]
        slice_data = group[1]
        slice_metrics = compute_model_metrics(slice_data[true_column], slice_data[predicted_column])
        slice_metrics["group_size"]=len(slice_data)
        performance_group[slice_name]=slice_metrics
        
    return performance_group