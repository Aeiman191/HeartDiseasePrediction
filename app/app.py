from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Assuming model.pkl is your trained model file
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({
        'Prediction': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
