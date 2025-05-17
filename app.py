from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS

model = load_model("diabetes_cnn_model.keras")
scaler = joblib.load("scaler.save")

@app.route('/')
def home():
    return 'Welcome to the Diabetes Prediction API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_reshaped = features_scaled.reshape(features_scaled.shape[0], features_scaled.shape[1], 1)
        prediction = model.predict(features_reshaped)
        result = int(prediction[0][0] > 0.5)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
