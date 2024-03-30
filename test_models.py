"""
Unit tests for the machine learning models.
"""
import unittest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestModels(unittest.TestCase):
    """
    Test case for the machine learning models.
    """
    def setUp(self):
        # Load test data
        self.test_data = pd.read_csv('heart_V2_test.csv')
        # Load trained models
        self.dt_model = joblib.load('DecisionTreeClassifier_model.pkl')
        self.svc_model = joblib.load('SVC_model.pkl')
        self.svc_f_model = joblib.load('SVC_model_2.pkl')
        self.lr_model = joblib.load('logistic_regression_model.pkl')

    def test_decision_tree_model(self):
        """
        Test decision tree model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']
        y_pred = self.dt_model.predict(x_test)
        self.assertGreaterEqual(accuracy_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(precision_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(recall_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(f1_score(y_test, y_pred), 0.50)


    def test_svc_model(self):
        """
        Test SVC model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']
        y_pred = self.svc_model.predict(x_test)
        self.assertGreaterEqual(accuracy_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(precision_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(recall_score(y_test, y_pred), 0.40)
        self.assertGreaterEqual(f1_score(y_test, y_pred), 0.40)


    def test_svc_f_model(self):
        """
        Test SVC model with linear kernel.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']
        y_pred = self.svc_f_model.predict(x_test)
        self.assertGreaterEqual(accuracy_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(precision_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(recall_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(f1_score(y_test, y_pred), 0.50)


    def test_logistic_regression_model(self):
        """
        Test logistic regression model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']
        y_pred = self.lr_model.predict(x_test)
        self.assertGreaterEqual(accuracy_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(precision_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(recall_score(y_test, y_pred), 0.50)
        self.assertGreaterEqual(f1_score(y_test, y_pred), 0.50)


if __name__ == '__main__':
    unittest.main()
