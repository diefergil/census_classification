# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

– Creator: Diego Fernández Gil
– Date: 2022-01-18
– Model version: 1
– Model type: Random Forest Classifier

## Intended Use

Educational purpose. Explore tools and methodologies for developing a CI/CD solution for a machine learning model.
## Training Data

The raw data used to train the model, along with more information about the features, can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

## Evaluation Data

The evaluation data (test) has been selected using a random selector and represents the 20% of the total data.

## Metrics

* Best params in grid_search: {'max_depth': 18, 'n_estimators': 800}
* Train metrics: {'precision': 0.8922475837960107, 'recall': 0.6947958366693354, 'fbeta': 0.7812387468491178}
* Test metrics: {'precision': 0.792046396023198, 'recall': 0.5989974937343359, 'fbeta': 0.6821262932572244}
