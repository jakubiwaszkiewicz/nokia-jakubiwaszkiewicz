from flask import Flask, request, jsonify
import joblib
import numpy as np
import bentoml

# Load the models
mlp_model = joblib.load("mlp.pkl")
knn_model = joblib.load("knn.pkl")
gnb_model = joblib.load("gnb.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the ML Prediction API!"

@app.route("/knn", methods=["POST"])
def knn():
    # Parse the JSON input
    data = request.get_json()
    input_data = np.array(data["features"]).reshape(1, -1)

    # Predict using each model
    knn_pred = knn_model.predict(input_data)

    # Combine the predictions into a response
    response = {
        "Prediction": knn_pred.tolist()
    }
    return jsonify(response)

@app.route("/gnb", methods=["POST"])
def gnb():
    # Parse the JSON input
    data = request.get_json()
    input_data = np.array(data["features"]).reshape(1, -1)

    # Predict using each model
    gnb_pred = gnb_model.predict(input_data)

    # Combine the predictions into a response
    response = {
        "Prediction": gnb_pred.tolist(),
    }
    return jsonify(response)

@app.route("/mlp", methods=["POST"])
def mlp():
    # Parse the JSON input
    data = request.get_json()
    input_data = np.array(data["features"]).reshape(1, -1)

    # Predict using each model
    mlp_pred = mlp_model.predict(input_data)

    # Combine the predictions into a response
    response = {
        "Prediction": mlp_pred.tolist(),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
