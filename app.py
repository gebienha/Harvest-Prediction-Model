from flask import Flask, render_template, request
import numpy as np
import webbrowser
import joblib
import pandas as pd
import scipy.stats as stats

# Load the model columns
model_columns = np.load('model_columns.npy', allow_pickle=True)

# Load the complete model with scaler
saved_model = joblib.load('harvest_model_complete.joblib')
model = saved_model['model']
scaler = saved_model['scaler']

# Load Box-Cox lambda
boxcox_lambda = joblib.load('boxcox_lambda.save')

# Make predictions on new data
def predict_crop_damage(new_data):
    # Apply Box-Cox transformation
    new_data['Estimated_Insects_Counts'] = stats.boxcox(new_data['Estimated_Insects_Count'], boxcox_lambda)
    new_data = new_data.drop(columns=['Estimated_Insects_Count'])
    # One-hot encode
    input_df = pd.get_dummies(new_data, columns=["Season", "Pesticide_Use_Category", "Soil_Type", "Crop_Type"])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict
    return model.predict(input_scaled)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    features = {
        "Estimated_Insects_Count": float(request.form["Estimated Insects Count"]),
        "Crop_Type": float(request.form["Crop Type"]),
        "Soil_Type": float(request.form["Soil_Type"]),
        "Pesticide_Use_Category": float(request.form["Pesticide Use Category"]),
        "Number_Doses_Week": float(request.form["Number Doses Week"]),
        "Number_Weeks_Used": float(request.form["Number Weeks Used"]),
        "Number_Weeks_Quit": float(request.form["Number Weeks Quit"]),
        "Season": float(request.form["Season"]),
    }
    # Convert to DataFrame
    input_df = pd.DataFrame([features])
    # One-hot encode
    input_df = pd.get_dummies(input_df, columns=["Season", "Pesticide_Use_Category", "Soil_Type", "Crop_Type"])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict - XGBoost returns the class directly
    prediction = predict_crop_damage(pd.DataFrame([features]))[0]
    pred_text = f"Harvest Prediction: Crop Damage Class {prediction}"
    return render_template("result.html", prediction_text=pred_text)

if __name__ == "__main__":
    app.run(debug=True)

webbrowser.open("http://127.0.0.1:5000/")

