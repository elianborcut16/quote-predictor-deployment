from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import logging
import os
import re
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

# Define our own Dutch stopwords instead of using NLTK
dutch_stopwords = {
    "aan", "achter", "af", "al", "alle", "alleen", "als", "altijd", "andere", "anders", 
    "ben", "bij", "daar", "dan", "dat", "de", "der", "deze", "die", "dit", "doch", "doen", 
    "door", "dus", "een", "eens", "eerder", "en", "enkele", "er", "eerst", "ge", "geen", 
    "geweest", "haar", "had", "hadden", "heb", "hebben", "heeft", "hem", "het", "hier", 
    "hij", "hoe", "hun", "iemand", "iets", "ik", "in", "is", "ja", "je", "kan", "kon", 
    "kunnen", "maar", "me", "meer", "men", "met", "mij", "mijn", "moet", "na", "naar", 
    "niet", "niets", "nog", "nu", "of", "om", "omdat", "onder", "ons", "ook", "op", "over", 
    "reeds", "te", "tegen", "toch", "toen", "tot", "u", "uit", "uw", "van", "veel", "voor", 
    "want", "waren", "was", "wat", "we", "wel", "werd", "wezen", "wie", "wij", "wil", "worden", 
    "wordt", "zal", "ze", "zelf", "zich", "zij", "zijn", "zo", "zonder", "zou"
}

# Define the custom analyzer functions needed by the vectorizer
# These must be defined at the module level

# Person-related terms for exclusion
person_terms = ["pax", "personen", "persoon", "mensen", "gasten", "gast", "pers"]

# This is a modified version of the function from the notebook
# We use our own stopwords set instead of NLTK
def should_exclude_word(word):
    """Check if word should be excluded based on fuzzy matching to domain terms."""
    # Skip short words
    if len(word) < 3:
        return True
    
    # Skip number-only words or words that start with numbers
    if word.isdigit() or (len(word) > 0 and word[0].isdigit()):
        return True
    
    # Check against Dutch stopwords
    if word.lower() in dutch_stopwords:
        return True
    
    # Check against person terms
    if process.extractOne(word, person_terms, scorer=fuzz.ratio)[1] > 75:
        return True
    
    return False

# This is the exact function from the notebook
def custom_analyzer(text):
    words = text.split()
    return [word for word in words if not should_exclude_word(word)]

from feature_extractor import FeatureExtractor

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
    model_path = os.path.join(os.path.dirname(__file__), "best_model_xgboost.pkl")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    # Create a TF-IDF vectorizer using saved vocabulary data
    logger.info("Creating TF-IDF vectorizer from saved vocabulary...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import json
    
    # Load the saved vocabulary and parameters from JSON
    vectorizer_data_path = os.path.join(os.path.dirname(__file__), "vectorizer_data.json")
    logger.info(f"Loading vectorizer data from {vectorizer_data_path}")
    
    try:
        with open(vectorizer_data_path, 'r') as f:
            vectorizer_data = json.load(f)
        
        # Create a new vectorizer with the same analyzer
        tfidf_vectorizer = TfidfVectorizer(
            analyzer=custom_analyzer,
            max_features=vectorizer_data.get('max_features')
        )
        
        # Convert vocabulary from strings back to integers
        # (JSON stores dictionary keys as strings, but TfidfVectorizer needs integer indices)
        vocabulary = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                      for k, v in vectorizer_data['vocabulary'].items()}
        
        # Don't use fit() since it resets the vocabulary
        # Instead, manually set up the attributes needed for transform()
        
        # Set the vocabulary directly
        tfidf_vectorizer.vocabulary_ = vocabulary
        
        # Create the expected shape for idf array
        expected_features = len(vocabulary)
        
        # Ensure idf array has the right shape
        if len(vectorizer_data['idf']) != expected_features:
            logger.warning(f"IDF array size ({len(vectorizer_data['idf'])}) doesn't match vocabulary size ({expected_features})")
            # Use default idf values (all 1's) if sizes don't match
            tfidf_vectorizer.idf_ = np.ones(expected_features)
        else:
            tfidf_vectorizer.idf_ = np.array(vectorizer_data['idf'])
            
        # We need to set up the _tfidf attribute and mark it as fitted
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer(norm='l2')
        transformer._idf_diag = None  # This needs to be set by fit
        
        # Create a dummy document matrix (1x1) with a single count of 1
        # This will initialize the transformer with default values
        X = np.array([[1.0]])
        transformer.fit(X)
        
        # Assign the fitted transformer
        tfidf_vectorizer._tfidf = transformer
        
        # Set stop words if they exist in the data
        if 'stop_words' in vectorizer_data and vectorizer_data['stop_words']:
            tfidf_vectorizer.stop_words_ = set(vectorizer_data['stop_words'])
        
        logger.info(f"TF-IDF vectorizer created successfully with {len(vocabulary)} terms")
    except Exception as e:
        logger.error(f"Failed to create TF-IDF vectorizer from JSON: {str(e)}")
        raise RuntimeError(f"Could not create TF-IDF vectorizer: {str(e)}")
    
    # Initialize the feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(tfidf_vectorizer=tfidf_vectorizer)
    logger.info("Feature extractor initialized")

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
    
        if "features" not in data:
            return jsonify({"error": "No features provided"}), 400
    
        features = np.array(data["features"]).reshape(1, -1)
    
        prediction = model.predict(features)[0]
        probas = model.predict_proba(features)[0]

        # Select correct confidence
        if prediction == 0:
            confidence = int(round(probas[0] * 100))
        else:
            confidence = int(round(probas[1] * 100))
    
        return jsonify({
            "prediction": int(prediction),
            "confidence": confidence
     })

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


    @app.route("/health", methods=["GET"])
    def health():
        # Check if tfidf_vectorizer is initialized
        tfidf_info = "loaded"
        if not hasattr(tfidf_vectorizer, 'vocabulary_'):
            tfidf_info = "error: vocabulary not found"
        
        return jsonify({
            "status": "healthy", 
            "model": "best_model_xgboost.pkl",
            "feature_extractor": "initialized",
            "tfidf_vectorizer": tfidf_info,
            "tfidf_vocabulary_size": len(tfidf_vectorizer.vocabulary_) if hasattr(tfidf_vectorizer, 'vocabulary_') else 0
        })

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)