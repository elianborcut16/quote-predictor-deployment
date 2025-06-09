# SAIL Prediction Model Deployment

This project contains the AI model deployment for the SAIL Dashboard, which predicts whether quotes will convert to bookings.

## Project Structure

- `app.py` - Flask application that serves the ML model
- `SAILmodel.pkl` - Trained scikit-learn model for predictions
- `feature_extractor/` - Python package for feature extraction
  - `extractor.py` - Direct implementation of the notebook's feature extraction code
  - `data/` - Directory for reference data files
    - `accounts.xlsx` - Company/account information
    - `ships.xlsx` - Ship information
    - `bookings_2015.xlsx` - Historical bookings data from 2015
    - `orders.xlsx` - Orders data
- `test_extractor.py` - Test script for the feature extractor

## API Endpoints

The Flask app provides the following endpoints:

### `/predict` (POST)

Makes a prediction for a quote. Accepts two types of input:

1. Quote data (JSON object with quote details)
2. Pre-extracted features (JSON object with "features" key)

Example request with quote data:
```json
{
  "quoteId": "Q12345",
  "customer": "ABN AMRO Bank N.V.",
  "type": "Standard",
  "packageName": "Sail in City",
  "subject": "Team event on Friday afternoon",
  "requestDate": "2023-10-01",
  "eventDate": "2023-11-15",
  "groupSize": 25,
  "isAmsterdam": true,
  "isRecurring": true
}
```

Example response:
```json
{
  "prediction": 1,
  "confidence": 87,
  "metadata": {
    "customer": "ABN AMRO Bank N.V.",
    "detected_ship": "sailboa",
    "detected_package": "sail_in_city",
    "detected_time": "middag",
    "detected_day": "vrijdag",
    "is_amsterdam": true,
    "is_recurring": true,
    "within_conversion_window": true,
    "feature_count": 49
  }
}
```

### `/transform` (POST)

Transforms a quote into feature vectors for the ML model. Accepts a quote object (same format as `/predict`).

Example response:
```json
{
  "features": [2023.0, 10.0, 1.0, 25.0, 7.0, ...],  
  "feature_dict": {
    "Created Year": 2023.0,
    "Created Month": 10.0,
    "Created DayOfWeek": 1.0,
    "Estimated group size": 25.0,
    "Subject_Word_Count": 7.0,
    "ship_sailboa": 1.0,
    "ship_jeroen_krabbé": 0.0,
    ...
  },
  "metadata": {
    "customer": "ABN AMRO Bank N.V.",
    "detected_ship": "sailboa",
    "detected_package": "sail_in_city",
    "detected_time": "middag",
    "detected_day": "vrijdag",
    "is_amsterdam": true,
    "is_recurring": true,
    "within_conversion_window": true,
    "feature_count": 49
  }
}
```

### `/health` (GET)

Health check endpoint.

Example response:
```json
{
  "status": "healthy",
  "model": "SAILmodel.pkl",
  "feature_extractor": "initialized"
}
```

## Reference Data

The feature extractor uses the exact same logic as the Jupyter notebook, with reference data from Excel files:

1. `accounts.xlsx` - Company information (city location, etc.)
2. `bookings_2015.xlsx` - Historical bookings data (to identify recurring customers)
3. `orders.xlsx` - Orders data for additional recurring customer detection

The feature extractor contains hardcoded data from the notebook:

- The complete list of ~800 ships from the notebook is included directly in the code
- Package variants and keywords are hardcoded
- Day and time keywords are hardcoded

## Running the Application

To run the Flask application:

```bash
python app.py
```

To test the feature extractor:

```bash
python test_extractor.py
```

## Implementation Notes

- The feature extraction logic is a direct implementation of the notebook code
- When a quote is submitted, it's transformed into a DataFrame and processed exactly as in the notebook
- Feature vectors are generated in the exact same order as the model expects
- No CSV or Excel files are saved - the features are returned directly through the API