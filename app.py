from flask import Flask, render_template, request
import numpy as np
import webbrowser
import joblib
import pandas as pd

# Load the model columns
model_columns = np.load('model_columns.npy', allow_pickle=True)

# Load the complete model with scaler
saved_model = joblib.load('harvest_model_complete.joblib')
model = saved_model['model']
scaler = saved_model['scaler']

# Make predictions on new data
def predict_crop_damage(new_data):
    # Preprocess new data using saved scaler
    scaled_data = scaler.transform(new_data)
    # Make prediction
    return model.predict(scaled_data)

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
    input_df = pd.get_dummies(input_df, columns=["Season", "Pesticide Use Category", "Soil_Type", "Crop Type"])
    # Reindex to match training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict - XGBoost returns the class directly
    prediction = model.predict(input_scaled)[0]
    pred_text = f"Harvest Prediction: Crop Damage Class {prediction}"
    return render_template("result.html", prediction_text=pred_text)

if __name__ == "__main__":
    app.run(debug=True)

webbrowser.open("http://127.0.0.1:5000/")