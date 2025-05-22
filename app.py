from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser
import joblib
import pandas as pd

# Load ANN model, scaler, and columns
model = load_model("harvestcrop.h5")
scaler = joblib.load("scaler.save")
model_columns = np.load("model_columns.npy", allow_pickle=True)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    features = {
        "Estimated Insects Count": float(request.form["Estimated Insects Count"]),
        "Crop Type": float(request.form["Crop Type"]),
        "Soil_Type": float(request.form["Soil_Type"]),
        "Pesticide Use Category": float(request.form["Pesticide Use Category"]),
        "Number Doses Week": float(request.form["Number Doses Week"]),
        "Number Weeks Used": float(request.form["Number Weeks Used"]),
        "Number Weeks Quit": float(request.form["Number Weeks Quit"]),
        "Season": float(request.form["Season"]),
    }
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    # One-hot encode
    input_df = pd.get_dummies(input_df)
    # Reindex to match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    pred_text = f"Harvest Prediction: Crop Damage Class {predicted_class}"
    return render_template("result.html", prediction_text=pred_text)

if __name__ == "__main__":
    app.run(debug=True)

webbrowser.open("http://127.0.0.1:5000/")