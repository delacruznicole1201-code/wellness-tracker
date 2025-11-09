from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("mental_health_model.pkl")

# Label mapping
label_map = {
    0: "Normal",
    1: "Stress",
    2: "Burnout",
    3: "Anxiety",
    4: "Depression"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        inputs = data.get("inputs")
        
        # Validate inputs
        if not isinstance(inputs, list) or len(inputs) != 4:
            return jsonify({"error": "Provide 4 numeric inputs."}), 400
        
        features = np.array(inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        result_label = label_map.get(prediction, "Unknown Result")
        return jsonify({"prediction": result_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
