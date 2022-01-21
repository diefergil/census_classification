# Script to train machine learning model.
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml import model as ml_model
from config import DATA_DIR, MODEL_DIR, logger, CAT_FEATURES
from sklearn.ensemble import RandomForestClassifier
import joblib
# Add code to load in the data.
logger.info("Reading data from: %s", DATA_DIR)
data = pd.read_csv(DATA_DIR / "census_cleaned.csv")
cat_features = CAT_FEATURES
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
logger.info("Train size: %g", len(train))
logger.info("Test size: %g", len(test))
logger.info("categorical features: %s", cat_features)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
param_grid = { 
    'n_estimators': [500, 800, 1000],
    #'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [8, 12, 16, 18],
    #'criterion' :['gini', 'entropy']
}
model = ml_model.train_model(
    X_train, 
    y_train, 
    model=RandomForestClassifier(random_state=42),
    hyperparameters=param_grid
)

train_preds = ml_model.inference(model, X_train)
train_metrics = ml_model.compute_model_metrics(y_train, train_preds)
logger.info("Train metrics: %s", train_metrics)
test_preds = ml_model.inference(model, X_test)
test_metrics = ml_model.compute_model_metrics(y_test, test_preds)
logger.info("Test metrics: %s", test_metrics)

#slice performance
train["y_true"]=lb.transform(train["salary"]).ravel()
train["y_pred"]=train_preds

train_slice_performance = dict()
for group in cat_features:
    train_slice_performance[group]=ml_model.calculate_slice_performance(train, group, "y_true","y_pred" )

logger.info("Saving  model outputs in: %s", MODEL_DIR)
joblib.dump(model, MODEL_DIR / "model.pkl")
joblib.dump(encoder, MODEL_DIR / "encoder.pkl")
joblib.dump(lb, MODEL_DIR / "label_binarizer.pkl")
