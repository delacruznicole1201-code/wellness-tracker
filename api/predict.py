from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("mental_health_model.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expect: { "inputs": [A, P, S, C] }
        inputs = data.get("inputs")

        if not isinstance(inputs, list) or len(inputs) != 4:
            return jsonify({"error": "Provide 4 numeric inputs: Academic, Personal, Social, Career"}), 400

        try:
            numeric_inputs = [float(i) for i in inputs]
        except:
            return jsonify({"error": "All inputs must be numeric"}), 400

        # Make into dataframe
        feature_names = ["Academic_Avg", "Personal_Avg", "Social_Avg", "Career_Avg"]
        features_df = pd.DataFrame([numeric_inputs], columns=feature_names)

        # Prediction
        prediction = model.predict(features_df)[0]
        result_label = le.inverse_transform([prediction])[0]

        return jsonify({"prediction": result_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Mental Health Prediction API is running!"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
