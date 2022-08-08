# Credit Card Approval Prediction

This project aims to predict credit card approval decision based on a number of variables, including but not limited to gender, occupation, income, family status, etc.

## Data Exploration
The exploratory data analysis from the dataset is conducted in [data_exploration.ipynb](data_exploration.ipynb), with mostly univariate analysis, to understand the dataset prior to transformation.

Specifically, the completeness of the dataset is the main focus at this stage, because it heavily impacts what types of transformation are needed before we can train a prediction model.

## Data Transformation
Based on the observations from exploratory data analysis, the datasets are transformed with the following major steps:
- Encoding Categorial Data
- Imputation (Filling missing data)
- Feature Engineering (Labelling the data)
- Merging the two Datasets
- Preparing Training & Testing Data

The corresponding notebook is located at [data_transformation.ipynb](data_transformation.ipynb).

## Model Training
A series of classifiers have been trained, and compared in terms of several performance metrics, including but not limited to training accuracy, testing accuracy, precision, recall and f-score.

A quick summary of the experiment is shown as follows:

| | Training Accuracy | Testing Accuracy | Precision | Recall | F-score |
|-|---|----|----|----|----|
| Gradient Boost | 99.7% | 93.1% | 99.6% | 86.5% | 0.926 |
|Adaboost | 82.6% | 82.3% | 85.7% | 77.6% | 0.815|
|Decision Trees | 99.8% | 83.5% | 99.0% | 67.6% | 0.804|
|Random Forest | 99.8% | 82.3% | 99.5% | 65.0% | 0.786|
| Logistic Regression | 60.9% | 55.8% | 56.4% | 51.5%| 0.539|
|Naive Bayes | 58.1% | 55.5% | 56.0% | 51.4% | 0.536 |

The ROC curve, and the detailed implementation for each classifier can be found in [model_training.ipynb](model_training.ipynb).

An example to illustrate how to use the model can be found in [using_model.ipynb](/using_model.ipynb).

**TODO: A web application will be built to facilitate the process and allow users to use the model with GUI.**

## Reference
Data Source: [https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
