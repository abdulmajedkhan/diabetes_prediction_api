from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter to make requests

# Initialize Firebase Admin
if not firebase_admin._apps:
    firebase_credentials = {
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace(r'\n', '\n'),  # Fix newline
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
    }

    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# Load ML model
model = joblib.load("best_diabetes_model.pkl")

# Feature order (should match training)
feature_names = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden Weight Loss',
                 'Weakness', 'Polyphagia', 'Genital Thrush', 'Visual Blurring',
                 'Itching', 'Irritability', 'Delayed Healing', 'Partial Paresis',
                 'Muscle Stiffness', 'Alopecia', 'Obesity']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON format
        print("Received data:", data)

        # Extract data from the request
        name = data.get('name')
        contact = data.get('contact')
        address = data.get('address')
        features = data.get('features')  # Expecting a list of 16 numbers

        # Input validation
        if not (name and contact and address and isinstance(features, list) and len(features) == 16):
            return jsonify({'error': 'Invalid input. Ensure correct fields and feature length.'}), 400

        # Convert features to numpy array for prediction
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        # Prediction message based on model output
        prediction_message = f"{name} Patient Diagnosed with early-stage diabetes" if prediction == 1 else "Individual assessed as low-risk for diabetes"

        # Save the prediction and features to Firestore
        record = {
            'name': name,
            'contact': contact,
            'address': address,
            'prediction': prediction_message,
            'prediction_label': int(prediction)
        }

        # Add feature values to the Firestore record
        for i, feature in enumerate(feature_names):
            val = features[i]
            if feature == 'Age':
                record[feature] = val
            elif feature == 'Gender':
                record[feature] = "Male" if val == 1 else "Female"
            else:
                record[feature] = "Yes" if val == 1 else "No"

        # Add record to Firestore
        db.collection('diabetes_predictions').add(record)

        # Return response to Flutter
        return jsonify({
            'prediction': prediction_message,
            'label': int(prediction)
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route("/", methods=["GET"])
def hello():
    return "Diabetes Prediction API is Live!", 200

if __name__ == '__main__':
    app.run(debug=True)
