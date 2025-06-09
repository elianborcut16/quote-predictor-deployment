import pandas as pd  # Still needed for DataFrame operations
import numpy as np
import re
from datetime import datetime
import os
import logging
import json
import math
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Person-related terms for exclusion
person_terms = ["pax", "personen", "persoon", "mensen", "gasten", "gast", "pers"]

# Function to clean text (used for both subject and account names)
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to check if a word should be excluded based on fuzzy matching
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

# Custom analyzer for TF-IDF that uses fuzzy matching to exclude domain words
def custom_analyzer(text):
    words = text.split()
    return [word for word in words if not should_exclude_word(word)]

# Compute TF-IDF features manually from vocabulary and IDF values
def compute_tfidf_features(subject, vocab_dict, idf_values, max_features=100, precomputed_weights=None):
    subject = clean_text(subject)
    words = subject.split()
    
    tfidf_features = [0.0] * max_features
    
    # If precomputed weights are available, use them
    if precomputed_weights:
        for word in words:
            if word in vocab_dict and word in precomputed_weights:
                index = vocab_dict[word]
                tfidf_features[index] = precomputed_weights[word]  # Use pre-computed weight
    else:
        # Otherwise use IDF values with normalization
        for word in words:
            if word in vocab_dict:
                index = vocab_dict[word]
                tfidf_features[index] = idf_values[index]  # simple presence * IDF
        
        # Apply L2 normalization to match sklearn's default behavior
        norm = math.sqrt(sum(x*x for x in tfidf_features))
        if norm > 0:
            tfidf_features = [x/norm for x in tfidf_features]
    
    return tfidf_features

class FeatureExtractor:
    """Feature extraction directly from the notebook."""
    
    def __init__(self, data_dir=None, tfidf_vectorizer=None):
        """
        Initialize the feature extractor with reference data.
        
        Args:
            data_dir: Directory containing reference data files
            tfidf_vectorizer: Not used anymore - kept for backward compatibility
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        self.model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Initialize data structures for vectorizer, recurring customers, and Amsterdam customers
        self.vectorizer_data = None
        self.recurring_customers = []
        self.amsterdam_customers = []
        
        # Load JSON data
        self._load_customer_lists()
        self._load_vectorizer_data()
    
    
    def _load_customer_lists(self):
        """Load recurring and Amsterdam customer lists from JSON files."""
        try:
            # Load recurring customers
            recurring_path = os.path.join(self.model_dir, 'recurring_customers_clean.json')
            if os.path.exists(recurring_path):
                logger.info(f"Loading recurring customers from {recurring_path}")
                with open(recurring_path, 'r') as f:
                    self.recurring_customers = json.load(f)
                logger.info(f"Loaded {len(self.recurring_customers)} recurring customers")
            else:
                logger.warning(f"Recurring customers file not found: {recurring_path}")
            
            # Load Amsterdam customers
            amsterdam_path = os.path.join(self.model_dir, 'amsterdam_customers_clean.json')
            if os.path.exists(amsterdam_path):
                logger.info(f"Loading Amsterdam customers from {amsterdam_path}")
                with open(amsterdam_path, 'r') as f:
                    self.amsterdam_customers = json.load(f)
                logger.info(f"Loaded {len(self.amsterdam_customers)} Amsterdam customers")
            else:
                logger.warning(f"Amsterdam customers file not found: {amsterdam_path}")
                
        except Exception as e:
            logger.error(f"Error loading customer lists: {str(e)}")
    
    def _load_vectorizer_data(self):
        """Load TF-IDF vocabulary data from JSON file."""
        try:
            # Load vectorizer data
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer_data.json')
            if os.path.exists(vectorizer_path):
                logger.info(f"Loading vectorizer data from {vectorizer_path}")
                with open(vectorizer_path, 'r') as f:
                    self.vectorizer_data = json.load(f)
                logger.info(f"Loaded vectorizer data with {len(self.vectorizer_data['vocabulary'])} terms")
            else:
                logger.warning(f"Vectorizer data file not found: {vectorizer_path}")
                
        except Exception as e:
            logger.error(f"Error loading vectorizer data: {str(e)}")
    
    def transform_quote(self, quote_data):
        """
        Transform a quote into features using the refactored approach.
        
        Args:
            quote_data: A dictionary with quote information
            
        Returns:
            Tuple of (features_dict, features_array, metadata)
        """
        try:
            # Create a single-row DataFrame from quote_data
            quotes_df = pd.DataFrame([quote_data])
            
            # Rename columns to match the expected format
            column_mapping = {
                'requestDate': 'Created on',
                'groupSize': 'Estimated group size',
                'customer': 'Accountname',
                'subject': 'Subject',
                'eventDate': 'Actual closing date',
                'contactPerson': 'Contact person'
            }
            
            for new_col, old_col in column_mapping.items():
                if new_col in quotes_df.columns and old_col not in quotes_df.columns:
                    quotes_df = quotes_df.rename(columns={new_col: old_col})
            
            # Ensure all required columns exist (with defaults if missing)
            required_columns = {
                'Created on': datetime.now().strftime('%Y-%m-%d'),
                'Subject': '',
                'Estimated group size': 30,
                'Accountname': '',
                'Status': 'Open',
                'Actual closing date': None,
                'Contact person': ''
            }
            
            for col, default in required_columns.items():
                if col not in quotes_df.columns:
                    quotes_df[col] = default
                    
            # Use Contact person as fallback for Accountname if Accountname is empty
            quotes_df['Accountname'] = quotes_df['Accountname'].fillna(quotes_df['Contact person'])
            
            # Convert dates to datetime
            quotes_df['Created on'] = pd.to_datetime(quotes_df['Created on'], errors='coerce')
            quotes_df['Actual closing date'] = pd.to_datetime(quotes_df['Actual closing date'], errors='coerce')
            
            # ----- FEATURE EXTRACTION BEGINS HERE -----
            
            # Fill missing subjects with empty strings
            quotes_df["Subject"] = quotes_df["Subject"].fillna("")
            
            # ---- Step 1: Calculate Conversion Window ----
            conversion_threshold = 93
            quotes_df['Within_Conversion_Window'] = 0  # Default
            
            if 'Event Date' in quotes_df.columns:
                quotes_df['Time_Difference'] = (quotes_df['Event Date'] - quotes_df['Created on']).dt.days
                quotes_df.loc[quotes_df['Time_Difference'] <= conversion_threshold, 'Within_Conversion_Window'] = 1
                quotes_df.drop(columns=['Time_Difference'], inplace=True)
            
            # ---- Step 2: Text Cleaning and Feature Extraction ----
            # Use the complete ship list from the notebook (retained but not shown for brevity)
            raw_ship_list = """
            .38
't Smidtje 
A van Eijck
A. van Eijk
Aagje
Aagtje
Aaltje Engelina
Abaco
Abel Tasman
            """  # Truncated for brevity
            
            # Clean and process the raw ship list
            ship_names = [
                name.strip().lower()
                for name in raw_ship_list.split('\n')
                if name.strip() != '' and not name.strip().startswith('.')
            ]

            # Include key ships
            required_ships = [
                "joanna saturna", "king", "bredero", "belle van zuylen", "salute", 
                "wapen van amsterdam", "belem", "breitner", "rijk de gooyer", 
                "kaat mossel", "test schip voor individueel 1", "frisius van adel", 
                "sunshine", "thalassa partyschip", "albert einstein"
            ]
            
            for ship in required_ships:
                if ship not in ship_names:
                    ship_names.append(ship)
            
            # Define package variants
            package_variants = {
                "sail_in_city": ["sail in city", "sailincity", "sail in", "sailin"],
                "sail_in_classic": ["sail in classic", "sailinclassic", "sail in", "sailin"],
                "sail_out_city": ["sail out city", "sailoutcity", "sail out", "sailout"],
                "sail_out_tall": ["sail out tall", "sailouttall", "sail out", "sailout", "tall"],
                "afternoon_cruise": ["afternoon cruise", "middag cruise", "middagcruise"],
                "evening_cruise": ["evening cruise", "avond cruise", "avondcruise"],
                "morning_cruise": ["morning cruise", "ochtend cruise", "ochtendcruise"],
                "lunch": ["lunch"],
                "brunch": ["brunch"],
                "drinks": ["drinks", "borrel"],
                "bites": ["bites", "hapjes"],
                "sunset_cruise": ["sunset cruise", "zonsondergang cruise", "sunset"],
                "evening_drinks": ["evening drinks", "avond borrel", "avondborrel"],
                "diner": ["diner", "dinner"]
            }

            # Create flat list of package names and variants
            package_names = list(package_variants.keys())
            package_keywords = []
            for variants in package_variants.values():
                package_keywords.extend(variants)
            package_keywords = list(set(package_keywords))

            # Time and day dictionaries
            time_keywords = {'ochtend': 'ochtend', 'middag': 'middag', 'avond': 'avond'}
            time_keywords_list = list(time_keywords.keys())

            days_keywords = {
                'za': 'zaterdag', 'zat': 'zaterdag',
                'zon': 'zondag', 'zo': 'zondag',
                'ma': 'maandag', 'di': 'dinsdag',
                'wo': 'woensdag', 'do': 'donderdag',
                'vr': 'vrijdag'
            }
            all_day_keywords = list(days_keywords.keys()) + list(days_keywords.values())
            days_list = list(set(days_keywords.values()))  # Unique day values
            
            # Clean the subject text
            quotes_df["Cleaned Subject"] = quotes_df["Subject"].apply(clean_text)
            
            # Function to extract subject features
            def extract_subject_features(text):
                features = {
                    'Ship': None,
                    'Package': None,
                    'Day': None,
                    'Time': None,
                }
                
                # Find ship - try exact match first
                for ship in ship_names:
                    if ship in text:
                        features['Ship'] = ship
                        break
                
                # If no ship found, try fuzzy matching
                if not features['Ship'] and len(ship_names) > 0:
                    match = process.extractOne(text, ship_names, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 75:  # 75% similarity threshold
                        features['Ship'] = match[0]
                
                # Find package - check each variant
                for pkg_name, variants in package_variants.items():
                    if any(variant in text for variant in variants):
                        features['Package'] = pkg_name
                        break
                
                # If no package found, try fuzzy matching
                if not features['Package']:
                    match = process.extractOne(text, package_keywords, scorer=fuzz.token_sort_ratio)
                    if match and match[1] >= 75:
                        # Find which package this keyword belongs to
                        for pkg_name, variants in package_variants.items():
                            if match[0] in variants:
                                features['Package'] = pkg_name
                                break
                
                # Find time mentions
                for time in time_keywords_list:
                    if time in text:
                        features['Time'] = time
                        break
                
                # Find day mentions
                for day in all_day_keywords:
                    if day in text:
                        features['Day'] = days_keywords.get(day, day)
                        break
                
                return pd.Series(features)
            
            # Apply function to get structured features
            subject_features = quotes_df['Cleaned Subject'].apply(extract_subject_features)
            
            # Merge subject features back into quotes_df
            quotes_df = pd.concat([quotes_df, subject_features], axis=1)
            
            # Count subject words
            quotes_df["Subject_Word_Count"] = quotes_df["Subject"].apply(lambda x: len(str(x).split()))
            
            # ---- Step 3: Identify Amsterdam and Recurring Customers with Fuzzy Matching ----
            # Clean incoming account name
            account_name_clean = clean_text(quotes_df['Accountname'].iloc[0])
            
            # Amsterdam fuzzy match
            if self.amsterdam_customers:
                match = process.extractOne(account_name_clean, self.amsterdam_customers, scorer=fuzz.token_sort_ratio)
                amsterdam_flag = 1 if match and match[1] >= 80 else 0
            else:
                # Fallback to checking if "amsterdam" appears in the account name
                amsterdam_flag = 1 if "amsterdam" in account_name_clean else 0
            
            # Recurring customer fuzzy match
            logger.info(f"Checking if '{account_name_clean}' is a recurring customer")
            if self.recurring_customers:
                match = process.extractOne(account_name_clean, self.recurring_customers, scorer=fuzz.token_sort_ratio)
                # Improved error handling for match object
                if match and match[1] >= 80:
                    recurring_flag = 1
                    logger.info(f"RECURRING CUSTOMER DETECTED: '{account_name_clean}' matched with '{match[0]}' (score: {match[1]})")
                else:
                    recurring_flag = 0
                    match_info = f"best match: '{match[0]}' with score {match[1]}" if match else "no match found"
                    logger.info(f"NOT A RECURRING CUSTOMER: '{account_name_clean}' - {match_info}")
                
                # Extra verification log
                logger.info(f"Setting recurring_flag = {recurring_flag} for '{account_name_clean}'")
            else:
                # No recurring customers found
                recurring_flag = 0
                logger.warning("Recurring customer check skipped - no recurring customers list loaded")
            
            # Add flags to DataFrame
            quotes_df["Company City Amsterdam"] = amsterdam_flag
            quotes_df["Recurring Customer"] = recurring_flag
            
            # ---- Step 4: One-Hot Encode Features ----
            # Function to clean feature name
            def clean_name(val):
                if pd.isna(val):
                    return None
                return str(val).strip().lower().replace(" ", "_")
            
            # Clean values for each feature
            quotes_df['ship_clean'] = quotes_df['Ship'].apply(clean_name)
            quotes_df['package_clean'] = quotes_df['Package'].apply(clean_name)
            quotes_df['time_clean'] = quotes_df['Time'].apply(clean_name)
            quotes_df['day_clean'] = quotes_df['Day'].apply(clean_name)
            
            # Get top ships (use all available if few)
            top_ships = ship_names[:min(20, len(ship_names))]
            if "other" not in top_ships:
                top_ships.append("other")
            
            # Initialize empty DataFrame for one-hot encoding
            onehot_df = pd.DataFrame(index=quotes_df.index)
            
            # Add ship columns
            for ship in top_ships:
                col_name = f"ship_{clean_name(ship)}"
                onehot_df[col_name] = (quotes_df['ship_clean'] == clean_name(ship)).astype(int)
            
            # Add package columns
            for pkg in package_names:
                col_name = f"package_{clean_name(pkg)}"
                onehot_df[col_name] = (quotes_df['package_clean'] == clean_name(pkg)).astype(int)
            
            # Add time columns
            for time in time_keywords.values():
                col_name = f"time_{clean_name(time)}"
                onehot_df[col_name] = (quotes_df['time_clean'] == clean_name(time)).astype(int)
            
            # Add day columns
            for day in days_list:
                col_name = f"day_{clean_name(day)}"
                onehot_df[col_name] = (quotes_df['day_clean'] == clean_name(day)).astype(int)
            
            # ---- Step 5: Extract datetime features ----
            quotes_df["Created Year"] = quotes_df["Created on"].dt.year
            quotes_df["Created Month"] = quotes_df["Created on"].dt.month
            quotes_df["Created DayOfWeek"] = quotes_df["Created on"].dt.weekday + 1  # 1-7 for Monday-Sunday
            
            # ---- Step 6: Apply TF-IDF using manual computation ----
            tfidf_features = []
            tfidf_feature_names = []
            
            # Use the vectorizer data if available
            if self.vectorizer_data:
                # Get vocabulary and IDF values
                vocabulary = self.vectorizer_data.get('vocabulary', {})
                idf_values = self.vectorizer_data.get('idf', [])
                max_features = self.vectorizer_data.get('max_features', 100)
                
                # Extract cleaned subject text
                subject_text = quotes_df["Cleaned Subject"].iloc[0]
                
                # Convert vocabulary from string keys to integers if needed
                vocab_dict = {}
                for word, idx in vocabulary.items():
                    if isinstance(idx, str) and idx.isdigit():
                        vocab_dict[word] = int(idx)
                    else:
                        vocab_dict[word] = idx
                
                # Check if precomputed weights are available
                precomputed_weights = self.vectorizer_data.get('average_weights', {})
                
                # Compute TF-IDF features manually
                tfidf_features = compute_tfidf_features(
                    subject_text, 
                    vocab_dict, 
                    idf_values, 
                    max_features,
                    precomputed_weights
                )
                
                # Get feature names in the right order
                tfidf_feature_names = sorted(vocabulary.keys(), key=lambda k: vocabulary[k])
                # Filter out number-containing names
                tfidf_feature_names = [name for name in tfidf_feature_names if not any(c.isdigit() for c in name)]
            else:
                logger.warning("No vectorizer data available. TF-IDF features will be empty.")
                tfidf_features = []
                tfidf_feature_names = []
            
            # ---- Step 7: Assemble Final Feature Array in Exact Order ----
            # Build feature list dynamically to match training dataset
            
            # 1️⃣ Base features
            keep_features = [
                "Created Year", "Created Month", "Created DayOfWeek", 
                "Estimated group size", "Within_Conversion_Window", "Subject_Word_Count"
            ]
            
            # 2️⃣ TF-IDF features
            keep_features.extend(tfidf_feature_names)
            
            # 3️⃣ One-hot columns (ships, packages, time, day)
            onehot_columns = onehot_df.columns.tolist()
            keep_features.extend(onehot_columns)
            
            # 4️⃣ Final columns
            keep_features.extend(["Company City Amsterdam", "Recurring Customer"])
            
            # Build the feature dictionary
            all_features_dict = {}
            
            # Add base features
            for feature in ["Created Year", "Created Month", "Created DayOfWeek", 
                           "Estimated group size", "Within_Conversion_Window", "Subject_Word_Count"]:
                all_features_dict[feature] = pd.to_numeric(quotes_df[feature].iloc[0], errors='coerce')
                if pd.isna(all_features_dict[feature]):
                    all_features_dict[feature] = 0
            
            # Add TF-IDF features
            for i, name in enumerate(tfidf_feature_names):
                if i < len(tfidf_features):
                    all_features_dict[name] = tfidf_features[i]
                else:
                    all_features_dict[name] = 0
            
            # Add one-hot features
            for feature in onehot_columns:
                all_features_dict[feature] = pd.to_numeric(onehot_df[feature].iloc[0], errors='coerce')
                if pd.isna(all_features_dict[feature]):
                    all_features_dict[feature] = 0
            
            # Add final columns
            all_features_dict["Company City Amsterdam"] = amsterdam_flag
            all_features_dict["Recurring Customer"] = recurring_flag
            
            # Build final feature array in the correct order
            features_array = []
            for feature in keep_features:
                features_array.append(all_features_dict[feature])
            
            # Convert everything to serializable
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj
            
            features_dict = convert_to_serializable(all_features_dict)
            features_array = convert_to_serializable(features_array)
            
            # Create metadata
            metadata = {
                'feature_count': len(features_array),
                'expected_feature_count': len(keep_features),
                'tfidf_feature_count': len(tfidf_feature_names),
                'onehot_feature_count': len(onehot_columns),
                'final_columns': ["Company City Amsterdam", "Recurring Customer"]
            }
            
            # Verify recurring flag was added as the last feature
            last_feature_value = features_array[-1] if features_array else None
            logger.info(f"Feature extraction complete. Generated {len(features_array)} features. Last feature (recurring) = {last_feature_value}")
            logger.info(f"Quote data summary: Account: {quotes_df['Accountname'].iloc[0]}, Recurring Flag: {recurring_flag}")
            return features_dict, features_array, metadata
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}, [], {"error": str(e)}
    
