# Script to train machine learning model.
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml import model as ml_model
from config import DATA_DIR, MODEL_DIR
from sklearn.ensemble import RandomForestClassifier
import joblib
# Add code to load in the data.
data = pd.read_csv(DATA_DIR / "census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = ml_model.train_model(X_train, y_train, model=RandomForestClassifier())

train_preds = ml_model.inference(model, X_train)
train_metrics = ml_model.compute_model_metrics(y_train, train_preds)
test_preds = ml_model.inference(model, X_test)
test_metrics = ml_model.compute_model_metrics(y_test, test_preds)

#slice performance
train["y_true"]=lb.transform(train["salary"]).ravel()
train["y_pred"]=train_preds

train_slice_performance = dict()
for group in cat_features:
    train_slice_performance[group]=ml_model.calculate_slice_performance(train, group, "y_true","y_pred" )


joblib.dump(model, MODEL_DIR / "model.pkl")
