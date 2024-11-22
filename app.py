from flask import Flask, request, jsonify
import joblib
import numpy as np

mlp_model = joblib.load("mlp.pkl")
knn_model = joblib.load("knn.pkl")
gnb_model = joblib.load("gnb.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Welcome to the ML Prediction API!</h1>
    <p>This API allows you to get predictions from three machine learning models: KNN, Gaussian Naive Bayes (GNB), and Multi-Layer Perceptron (MLP).</p>
    <h2>How to Use</h2>
    <ol>
        <li>Send a POST request to the following endpoints:</li>
        <ul>
            <li><strong>/knn</strong> - For predictions using the KNN model.</li>
            <li><strong>/gnb</strong> - For predictions using the Gaussian Naive Bayes model.</li>
            <li><strong>/mlp</strong> - For predictions using the Multi-Layer Perceptron model.</li>
        </ul>
        <li>The POST request should contain JSON data in the following format (below is already sample data which you can send on the backend server):</li>
        <pre>{
    "features": [5.0, 13.0, 9.0, 1.0, 0.0, 0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0]
}</pre>
        <li>You will receive a JSON response with the model's prediction.</li>
    </ol>
    <p>Example using <strong>curl</strong>:</p>
    <span>curl -X POST https://nokia-jakubiwaszkeiwicz-recruitment-task.azurewebsites.net/knn -H "Content-Type: application/json" -d '{"features": [5.0, 13.0, 9.0, 1.0, 0.0, 0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0]}'</span>
    """

@app.route("/knn", methods=["POST"])
def knn():
    data = request.get_json()
    input_data = np.array(data["features"])

    knn_pred = knn_model.predict(input_data)

    response = {
        "Prediction": knn_pred
    }
    return jsonify(response)

@app.route("/gnb", methods=["POST"])
def gnb():
    data = request.get_json()
    input_data = np.array(data["features"])

    gnb_pred = gnb_model.predict(input_data)

    response = {
        "Prediction": gnb_pred
    }
    return jsonify(response)

@app.route("/mlp", methods=["POST"])
def mlp():
    data = request.get_json()
    input_data = np.array(data["features"]).reshape(1, -1)

    mlp_pred = mlp_model.predict(input_data)

    response = {
        "Prediction": mlp_pred
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
