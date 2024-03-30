"""
Unit tests for the machine learning models.
"""
import pickle
import unittest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestModels(unittest.TestCase):
    """
    Test case for the machine learning models.
    """
    def setUp(self):
        # Load test data
        self.test_data = pd.read_csv('test_data.csv')

        # Load trained models
        with open('DecisionTreeClassifier_model.pkl', 'rb') as file:
            self.dt_model = pickle.load(file)
        with open('SVC_model.pkl', 'rb') as file:
            self.svc_model = pickle.load(file)
        with open('SVC_model_2.pkl', 'rb') as file:
            self.svc_f_model = pickle.load(file)
        with open('logistic_regression_model.pkl', 'rb') as file:
            self.lr_model = pickle.load(file)

    def test_decision_tree_model(self):
        """
        Test decision tree model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.dt_model.predict(x_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.90, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.85, places=2)

    def test_svc_model(self):
        """
        Test SVC model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.svc_model.predict(x_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.70, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.75, places=2)

    def test_svc_f_model(self):
        """
        Test SVC model with linear kernel.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.svc_f_model.predict(x_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.80, places=2)

    def test_logistic_regression_model(self):
        """
        Test logistic regression model.
        """
        x_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.lr_model.predict(x_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.80, places=2)

if __name__ == '__main__':
    unittest.main()
