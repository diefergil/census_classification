from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from starter.ml.model import inference
from starter.config import MODEL_DIR, CAT_FEATURES, logger

# Load utils
model = joblib.load(MODEL_DIR / "model.pkl")
encoder = joblib.load(MODEL_DIR / "encoder.pkl")
label_binarizer = joblib.load(MODEL_DIR / "label_binarizer.pkl")
categorical_features=CAT_FEATURES


class InputData(BaseModel):
    """data definition to be used for predictions."""

    age: int = Field(..., example=68)
    workclass: str = Field(..., example="State-gov", )
    fnlgt: int = Field(..., example=146645)
    education: str = Field(..., example="Doctorate")
    education_num: int = Field(..., example=16, alias='education-num')
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Farming-fishing")
    relationship: str = Field(..., example="Unmarried")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain:int = Field(..., example=20051, alias="capital-gain")
    capital_loss:int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=50, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")
    
    #The allow_population_by_field_name option is needed to allow creating 
    # object with field name, without it you could instantiate it only with alias name.
    class Config:
        allow_population_by_field_name = True


def make_prediction(message, model, encoder):
    try:
        message = message.dict(by_alias=True)
        logger.info("message: %s", message)
        data = pd.DataFrame(message, index=[0, ])
        X_categorical = data[categorical_features].values
        X_continuous = data.drop(*[categorical_features], axis=1)
        X_categorical = encoder.transform(X_categorical)
        X = np.concatenate([X_continuous, X_categorical], axis=1)
        prediction = inference(model, X)
        prediction_str = label_binarizer.inverse_transform(prediction)[0]
        logger.info("prediction: %s", prediction_str)
    except Exception as e:
        logger.error("Error: %s", e)
        prediction_str="FAIL"
    
    return prediction_str


    
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/predict/')
async def predict(input_data: InputData):
    
    prediction = make_prediction(
        message=input_data, model=model, encoder=encoder
        )
    return {"prediction": prediction}

# Example
# from typing import Union, List
#class TaggedItem(BaseModel):
#   name: str
#   tags: Union[str, List[str]]
#   item_id: int
#   native_country: str = Field(..., example="United-States", alias="native-country")
#   class Config:
#       allow_population_by_field_name = True
#
# This allows sending of data (our TaggedItem) via POST to the API.
#app.post("/items/")
#sync def create_item(item: TaggedItem):
#   return item
#
#app.get("/items/{item_id}")
#sync def get_items(item_id: int, count: int = 1):
#   return {"fetch": f"Fetched {count} of {item_id}"}
#
#
#app.post("/{path}")
#sync def exercise_function(path: int, query: int, body: TaggedItem):
#   return {"path": path, "query": query, "body": body}
#
#app.get("/items/{item_id}")
#sync def get_items(item_id: int, count: int = 1):
#   return {"fetch": f"Fetched {count} of {item_id}"}
