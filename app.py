# from flask import Flask, render_template, request
# import numpy as np
# from tensorflow.keras.models import load_model
# import webbrowser
# import joblib
# import pandas as pd

# # Load ANN model, scaler, and columns
# model = load_model("harvestcrop.h5")
# scaler = joblib.load("scaler.save")
# model_columns = np.load("model_columns.npy", allow_pickle=True)

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get data from form
#     features = {
#         "Estimated Insects Count": float(request.form["Estimated Insects Count"]),
#         "Crop Type": float(request.form["Crop Type"]),
#         "Soil_Type": float(request.form["Soil_Type"]),
#         "Pesticide Use Category": float(request.form["Pesticide Use Category"]),
#         "Number Doses Week": float(request.form["Number Doses Week"]),
#         "Number Weeks Used": float(request.form["Number Weeks Used"]),
#         "Number Weeks Quit": float(request.form["Number Weeks Quit"]),
#         "Season": float(request.form["Season"]),
#     }
#     # Convert to DataFrame
#     input_df = pd.DataFrame([features])
#     # One-hot encode
#     input_df = pd.get_dummies(input_df)
#     # Reindex to match training columns
#     input_df = input_df.reindex(columns=model_columns, fill_value=0)
#     # Scale
#     input_scaled = scaler.transform(input_df)
#     # Predict
#     prediction = model.predict(input_scaled)
#     predicted_class = int(np.argmax(prediction, axis=1)[0])
#     pred_text = f"Harvest Prediction: Crop Damage Class {predicted_class}"
#     return render_template("result.html", prediction_text=pred_text)

# if __name__ == "__main__":
#     app.run(debug=True)

# webbrowser.open("http://127.0.0.1:5000/")

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
import webbrowser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
model_columns = None

def load_model_components():
    """Load and validate all model components"""
    global model, scaler, model_columns
    
    try:
        # Check if files exist
        required_files = ["harvestcrop.h5", "scaler.save", "model_columns.npy"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
            
        # Load model
        logger.info("Loading ANN model...")
        model = load_model("harvestcrop.h5")
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
        # Load scaler
        logger.info("Loading scaler...")
        scaler = joblib.load("scaler.save")
        logger.info(f"Scaler loaded successfully. Feature names: {getattr(scaler, 'feature_names_in_', 'Not available')}")
        
        # Load model columns
        logger.info("Loading model columns...")
        model_columns = np.load("model_columns.npy", allow_pickle=True)
        logger.info(f"Model columns loaded: {len(model_columns)} features")
        logger.info(f"Columns: {list(model_columns)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def validate_input_data(features):
    """Validate input data types and ranges"""
    try:
        # Define expected ranges or types for validation
        validation_rules = {
            "Estimated Insects Count": {"min": 0, "max": 10000},
            "Crop Type": {"min": 0, "max": 100},  # Adjust based on your encoding
            "Soil_Type": {"min": 0, "max": 100},  # Adjust based on your encoding
            "Pesticide Use Category": {"min": 0, "max": 100},  # Adjust based on your encoding
            "Number Doses Week": {"min": 0, "max": 50},
            "Number Weeks Used": {"min": 0, "max": 52},
            "Number Weeks Quit": {"min": 0, "max": 52},
            "Season": {"min": 0, "max": 4}  # Assuming seasons are encoded 0-3
        }
        
        for feature, value in features.items():
            if feature in validation_rules:
                min_val = validation_rules[feature]["min"]
                max_val = validation_rules[feature]["max"]
                if not (min_val <= value <= max_val):
                    logger.warning(f"{feature} value {value} is outside expected range [{min_val}, {max_val}]")
        
        return True
    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        return False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Received prediction request")
        
        # Check if model components are loaded
        if model is None or scaler is None or model_columns is None:
            logger.error("Model components not loaded properly")
            return render_template("result.html", 
                                 prediction_text="Error: Model not loaded properly",
                                 error=True)
        
        # Get data from form
        logger.info("Extracting features from form")
        features = {}
        
        # Get form data with error handling
        form_fields = [
            "Estimated Insects Count",
            "Crop Type", 
            "Soil_Type",
            "Pesticide Use Category",
            "Number Doses Week",
            "Number Weeks Used", 
            "Number Weeks Quit",
            "Season"
        ]
        
        for field in form_fields:
            try:
                value = float(request.form[field])
                features[field] = value
                logger.info(f"{field}: {value}")
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing field {field}: {str(e)}")
                return render_template("result.html",
                                     prediction_text=f"Error: Invalid input for {field}",
                                     error=True)
        
        # Validate input data
        if not validate_input_data(features):
            return render_template("result.html",
                                 prediction_text="Error: Input validation failed",
                                 error=True)
        
        # Convert to DataFrame
        logger.info("Converting to DataFrame")
        input_df = pd.DataFrame([features])
        logger.info(f"Input DataFrame shape: {input_df.shape}")
        logger.info(f"Input DataFrame columns: {list(input_df.columns)}")
        
        # Handle categorical encoding
        logger.info("Applying one-hot encoding")
        input_encoded = pd.get_dummies(input_df)
        logger.info(f"After encoding shape: {input_encoded.shape}")
        logger.info(f"After encoding columns: {list(input_encoded.columns)}")
        
        # Reindex to match training columns
        logger.info("Reindexing to match training columns")
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        logger.info(f"Final input shape: {input_final.shape}")
        logger.info(f"Expected features: {len(model_columns)}")
        
        # Check for any missing columns
        missing_cols = set(model_columns) - set(input_encoded.columns)
        if missing_cols:
            logger.info(f"Missing columns (filled with 0): {missing_cols}")
        
        # Scale the features
        logger.info("Scaling features")
        input_scaled = scaler.transform(input_final)
        logger.info(f"Scaled input shape: {input_scaled.shape}")
        logger.info(f"Scaled input sample: {input_scaled[0][:5]}...")  # Show first 5 values
        
        # Make prediction
        logger.info("Making prediction")
        prediction = model.predict(input_scaled)
        logger.info(f"Raw prediction shape: {prediction.shape}")
        logger.info(f"Raw prediction values: {prediction[0]}")
        
        # Get predicted class
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction, axis=1)[0])
        
        logger.info(f"Predicted class: {predicted_class}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        # Create prediction text with more details
        pred_text = f"Harvest Prediction: Crop Damage Class {predicted_class}"
        confidence_text = f"Confidence: {confidence:.2%}"
        
        return render_template("result.html", 
                             prediction_text=pred_text,
                             confidence_text=confidence_text,
                             error=False)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template("result.html",
                             prediction_text=f"Error during prediction: {str(e)}",
                             error=True)

@app.route("/debug")
def debug_info():
    """Debug endpoint to check model status"""
    info = {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_columns_loaded": model_columns is not None,
        "model_input_shape": str(model.input_shape) if model else "Not loaded",
        "model_output_shape": str(model.output_shape) if model else "Not loaded",
        "num_features": len(model_columns) if model_columns is not None else "Not loaded",
        "scaler_type": type(scaler).__name__ if scaler else "Not loaded"
    }
    
    if model_columns is not None:
        info["feature_columns"] = list(model_columns)
    
    return jsonify(info)

if __name__ == "__main__":
    # Load model components before starting the app
    logger.info("Starting Flask application...")
    
    if load_model_components():
        logger.info("All model components loaded successfully!")
        logger.info("Starting Flask server...")
        
        # Open browser after a short delay
        import threading
        import time
        
        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open("http://127.0.0.1:5000/")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        app.run(debug=True, use_reloader=False)  # Disable reloader to prevent duplicate browser opening
    else:
        logger.error("Failed to load model components. Please check your model files.")
        print("\nPlease ensure the following files exist in your project directory:")
        print("- harvestcrop.h5 (your trained model)")
        print("- scaler.save (your fitted scaler)")
        print("- model_columns.npy (your feature column names)")