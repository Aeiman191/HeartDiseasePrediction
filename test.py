# test_models.py

import pytest
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('heart_v2.csv')
data_test = df['heart disease']
data_train = df.drop('heart disease', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    data_train, data_test, test_size=0.2, random_state=42)

# Decision Tree Classifier
def test_decision_tree_classifier():
    with open('DecisionTreeClassifier_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    y_pred = clf.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= 0.7
    assert precision_score(y_test, y_pred) >= 0.7
    assert recall_score(y_test, y_pred) >= 0.7
    assert f1_score(y_test, y_pred) >= 0.7

# Logistic Regression Model
def test_logistic_regression_model():
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= 0.7
    assert precision_score(y_test, y_pred) >= 0.7
    assert recall_score(y_test, y_pred) >= 0.7
    assert f1_score(y_test, y_pred) >= 0.7

# Support Vector Machine Classifier
def test_svc_model():
    with open('SVC_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    y_pred = clf.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= 0.7
    assert precision_score(y_test, y_pred) >= 0.7
    assert recall_score(y_test, y_pred) >= 0.7
    assert f1_score(y_test, y_pred) >= 0.7

# Support Vector Machine Classifier with linear kernel
def test_svc_model_2():
    with open('SVC_model_2.pkl', 'rb') as file:
        clf = pickle.load(file)
    y_pred = clf.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= 0.7
    assert precision_score(y_test, y_pred) >= 0.7
    assert recall_score(y_test, y_pred) >= 0.7
    assert f1_score(y_test, y_pred) >= 0.7
