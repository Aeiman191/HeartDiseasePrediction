"""
This is a Flask application for serving a machine learning model.
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    """
    Home endpoint returning a welcome message.
    """
    return "Welcome to the Heart Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that receives input data and returns model predictions.
    """
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({
        'Prediction': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
