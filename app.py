from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging
import os
import json
from rapidfuzz import process, fuzz

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# Import the feature extractor
from feature_extractor import FeatureExtractor
from feature_extractor.extractor import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    app.json_encoder = NumpyJSONEncoder  # Use our custom JSON encoder
    CORS(app)

    # Load the ML model
    logger.info("Loading ML model...")
    model_path = os.path.join(os.path.dirname(__file__), "best_model_xgboost_calibrated.pkl")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    # Initialize the feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = FeatureExtractor()
    logger.info("Feature extractor initialized")

    def process_prediction(features):
        """
        Process predictions using the model - used by both /predict and /predict_batch
        
        Args:
            features: numpy array of features (1D or 2D)
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure features is a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Get raw probabilities
        probas = model.predict_proba(features)
        
        # Adjust threshold for class imbalance
        booking_threshold = 0.15  # Threshold for booking class
        
        results = []
        for i, proba in enumerate(probas):
            # Make prediction using custom threshold
            booking_probability = proba[1]  # Probability of being a booking (class 1)
            prediction = 1 if booking_probability >= booking_threshold else 0
            
            # Calculate confidence (0-100)
            if prediction == 1:  # Booking prediction
                # Boost booking confidence by applying a scaling factor
                # Justification: The model tends to be conservative with booking probabilities.
                # In reality, when the model predicts a booking, it's often correct but with low confidence.
                # This adjustment helps present more confident predictions to users when the model
                # identifies a potential booking, improving user trust and decision-making.
                
                # Map the range 0.20-0.50 to approximately 65-95 range
                # The min threshold is set to 65% to avoid showing "low confidence" bookings
                # The max is capped at 95% to maintain some uncertainty even in high-confidence cases
                min_booking_confidence = 65  # Minimum confidence for bookings
                max_booking_confidence = 95  # Maximum confidence for bookings
                
                # Calculate scaled confidence - linear scaling between thresholds
                if booking_probability >= 0.5:
                    # If probability is very high (≥0.5), give it max confidence
                    confidence = max_booking_confidence
                else:
                    # Scale the probability between booking_threshold and 0.5 to min_booking_confidence and max_booking_confidence
                    scale_factor = (max_booking_confidence - min_booking_confidence) / (0.5 - booking_threshold)
                    confidence = int(min_booking_confidence + (booking_probability - booking_threshold) * scale_factor)
                    # Ensure confidence stays within expected range
                    confidence = max(min_booking_confidence, min(max_booking_confidence, confidence))
            else:  # Non-booking prediction
                # For non-bookings, keep the original calculation
                confidence = int(round((1 - booking_probability) * 100))
            
            # Extract recurring customer flag (last feature for 0-based indexing)
            # The last feature in the array is the recurring customer flag
            is_recurring = "1" if features[i][-1] == 1 else "0"
            
            # Extract package information (features 128-141)
            package_indices = range(128, 142)  # 14 package features from 128 to 141 inclusive
            package_names = [
                "sail_in_city", "sail_in_classic", "sail_out_city", "sail_out_tall",
                "afternoon_cruise", "evening_cruise", "morning_cruise", "lunch",
                "brunch", "drinks", "bites", "sunset_cruise", "evening_drinks", "diner"
            ]
            
            # Get packages with value 1 (selected)
            packages = []
            for idx, pkg_name in zip(package_indices, package_names):
                if idx < len(features[i]) and features[i][idx] == 1:
                    packages.append(pkg_name)
                    logger.info(f"Row {i+1}: Package detected: {pkg_name} at feature index {idx}")
            
            # Log prediction details including recurring status and packages
            logger.info(f"Row {i+1}: Prediction: {prediction}, Booking probability: {booking_probability:.4f}, Confidence: {confidence}%, Recurring: {is_recurring}, Feature[-1]: {features[i][-1]}, Total features: {len(features[i])}, Packages: {packages}")
            
            # Add to results with prediction, confidence, recurring status, and packages
            results.append({
                "prediction": str(prediction),  # String format for consistency
                "confidence": confidence,       # Integer 0-100
                "recurring": is_recurring,      # String "1" or "0" for consistency
                "packages": packages            # List of package names detected
            })
            
        return results

    @app.route("/predict", methods=["POST"])
    def predict():
        # Forward to the batch prediction endpoint with a single row
        data = request.get_json()
        
        logger.info(f"Single prediction request received: {json.dumps(data)[:200]}...")
    
        if "features" not in data:
            return jsonify({"error": "No features provided"}), 400
    
        # Get the features array
        features = np.array(data["features"])
        
        # Convert to batch format if needed
        if features.ndim == 2 and len(features) == 1:
            features = features[0]
        
        # Get prediction results
        results = process_prediction(features)
        result = results[0]  # We only have one result for /predict
        
        # Log the result before returning
        response = {
            "prediction": [result["prediction"]],
            "confidence": [result["confidence"]],
            "recurring": [result["recurring"]],
        }
        logger.info(f"Single prediction response: {json.dumps(response)}")
        
        # Return in the requested format, including recurring customer flag
        return jsonify(response)

    @app.route("/transform", methods=["POST"])
    def transform():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Transform quote to features using the feature extractor
        _, features_array, _ = feature_extractor.transform_quote(data)

        return jsonify({
            "features": features_array,
            "feature_count": len(features_array)
        })

    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        data = request.get_json()
        
        if not data or "rows" not in data:
            return jsonify({"error": "No data provided or missing 'rows' field"}), 400
        
        if not isinstance(data["rows"], list):
            return jsonify({"error": "'rows' must be an array of quote objects"}), 400
        
        if len(data["rows"]) == 0:
            return jsonify({"results": []}), 200
            
        # Log the batch size and sample of the first request
        logger.info(f"Processing batch prediction request with {len(data['rows'])} rows")
        if len(data["rows"]) > 0:
            logger.info(f"First row sample: {json.dumps(data['rows'][0])[:200]}...")
        
        # Process each row to extract features
        all_features = []
        for i, row in enumerate(data["rows"]):
            try:
                # Log the customer name for debugging purposes only
                account_name = None
                if "Accountname" in row:
                    account_name = row["Accountname"]
                elif "accountname" in row:
                    account_name = row["accountname"]
                
                if account_name:
                    # Just log the customer name for debugging - all processing happens in feature_extractor
                    logger.info(f"Row {i+1} - Processing customer: '{account_name}'")
                    if not feature_extractor.recurring_customers:
                        logger.warning("No recurring customers loaded!")
                
                # Transform to features
                _, features, _ = feature_extractor.transform_quote(row)
                all_features.append(features)
                
                # Log the last feature (recurring flag) and feature count
                logger.info(f"Processed row {i+1}/{len(data['rows'])} - Last feature (recurring): {features[-1]}, Total features: {len(features)}")
            except Exception as e:
                logger.error(f"Error processing row {i+1}: {str(e)}")
                return jsonify({"error": f"Error processing row {i+1}: {str(e)}"}), 400
        
        # Convert to numpy array for batch prediction
        features_array = np.array(all_features)
        
        # Get prediction results
        results = process_prediction(features_array)
        
        # Log final results before returning
        logger.info(f"Batch processing complete. Returning {len(results)} results")
        logger.info(f"Final results sample: {json.dumps(results[:min(3, len(results))])}")
        
        # Return batch results with prediction, confidence, and recurring flag
        return jsonify({
            "results": results
        })

    @app.route("/health", methods=["GET"])
    def health():
        # Check if the feature extractor is properly loaded
        vectorizer_info = "not loaded"
        vocabulary_size = 0
        
        if hasattr(feature_extractor, 'vectorizer_data') and feature_extractor.vectorizer_data:
            vectorizer_info = "loaded"
            vocabulary_size = len(feature_extractor.vectorizer_data.get('vocabulary', {}))
        
        # Check recurring customers and Amsterdam customers
        recurring_count = len(feature_extractor.recurring_customers)
        amsterdam_count = len(feature_extractor.amsterdam_customers)
        
        return jsonify({
            "status": "healthy", 
            "model": "best_model_xgboost.pkl",
            "feature_extractor": "initialized",
            "vectorizer_data": vectorizer_info,
            "vocabulary_size": vocabulary_size,
            "recurring_customers_count": recurring_count,
            "amsterdam_customers_count": amsterdam_count
        })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)