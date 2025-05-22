from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import webbrowser

# Load ANN model
model = load_model("harvestcrop.h5")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form in the correct order
    features = [
        float(request.form["Estimated Insects Count"]),
        float(request.form["Crop Type"]),
        float(request.form["Soil_Type"]),
        float(request.form["Pesticide Use Category"]),
        float(request.form["Number Doses Week"]),
        float(request.form["Number Weeks Used"]),
        float(request.form["Number Weeks Quit"]),
        float(request.form["Season"]),
    ]

    # Convert to numpy array for prediction
    input_array = np.array([features])  # Shape: (1, 8)
    
    prediction = model.predict(input_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])  # Get class 0, 1, or 2

    pred_text = f"Harvest Prediction: Crop Damage Class {predicted_class}"
    return render_template("result.html", prediction_text=pred_text)

if __name__ == "__main__":
    app.run(debug=True)

webbrowser.open("http://127.0.0.1:5000/")