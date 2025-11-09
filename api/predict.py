import json
import joblib
import numpy as np

# Load the trained model once (cold start)
model = joblib.load("mental_health_model.pkl")

# Label mapping
label_map = {
    0: "Normal",
    1: "stress",
    2: "burnout",
    3: "anxiety",
    4: "depression"
}

def handler(request):
    try:
        # Parse JSON body
        body = request.get_json()
        inputs = body.get("inputs")  # expects list of 4 numbers

        # Validate inputs
        if not isinstance(inputs, list) or len(inputs) != 4:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Provide 4 numeric inputs."})
            }

        # Make prediction
        features = np.array(inputs).reshape(1, -1)
        prediction = model.predict(features)[0]
        result_label = label_map.get(prediction, "Unknown Result")

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": result_label})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
