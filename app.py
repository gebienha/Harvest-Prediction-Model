from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the trained ANN model
model = load_model("NN_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Collect form data and convert to float
    float_inputs = [float(x) for x in request.form.values()]
    
    # Shape into (1, n_features) as required by Keras
    input_array = np.array([float_inputs])
    
    # Predict class probabilities
    prediction = model.predict(input_array)
    
    # Choose class with highest probability
    output_class = int(np.argmax(prediction))

    return render_template("result.html", prediction_text=f'Harvest Prediction: {output_class}')

if __name__ == "__main__":
    app.run(debug=True)
