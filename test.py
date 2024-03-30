import unittest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.externals import joblib
from models import predict_models

class TestModels(unittest.TestCase):
    def setUp(self):
        # Load test data
        self.test_data = pd.read_csv('test_data.csv')

        # Load trained models
        self.dt_model = joblib.load('DecisionTreeClassifier_model.pkl')
        self.svc_model = joblib.load('SVC_model.pkl')
        self.svc_f_model = joblib.load('SVC_model_2.pkl')
        self.lr_model = joblib.load('logistic_regression_model.pkl')

    def test_decision_tree_model(self):
        X_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.dt_model.predict(X_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.90, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.85, places=2)

    def test_svc_model(self):
        X_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.svc_model.predict(X_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.70, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.75, places=2)

    def test_svc_f_model(self):
        X_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.svc_f_model.predict(X_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.80, places=2)

    def test_logistic_regression_model(self):
        X_test = self.test_data.drop('heart disease', axis=1)
        y_test = self.test_data['heart disease']

        y_pred = self.lr_model.predict(X_test)

        self.assertAlmostEqual(accuracy_score(y_test, y_pred), 0.80, places=2)
        self.assertAlmostEqual(precision_score(y_test, y_pred), 0.75, places=2)
        self.assertAlmostEqual(recall_score(y_test, y_pred), 0.85, places=2)
        self.assertAlmostEqual(f1_score(y_test, y_pred), 0.80, places=2)

if __name__ == '__main__':
    unittest.main()
