from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import lightgbm as lgb
from tensorflow.keras.models import load_model
import joblib

location_le = joblib.load('location_le.pkl')
city_le = joblib.load('city_le.pkl')


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Allow frontend origin for CORS

# Load trained models and scaler
lgb_model = lgb.Booster(model_file='lgb_model.txt')
nn_model = load_model('nn_model.h5')
scaler = joblib.load('scaler.pkl')

# City templates (simplified for brevity - include all your keys)
mumbai_template = {
    "Price": 2500000,
    "Area": 774,
    "Location": "Andheri West",
    "No_of_Bedrooms": 2,
    "Resale": 1,
    "MaintenanceStaff": 9,
    "Gymnasium": 9,
    "SwimmingPool": 9,
    "LandscapedGardens": 9,
    "JoggingTrack": 9,
    "RainWaterHarvesting": 9,
    "IndoorGames": 9,
    "ShoppingMall": 9,
    "Intercom": 9,
    "SportsFacility": 9,
    "ATM": 9,
    "ClubHouse": 9,
    "School": 9,
    "24X7Security": 9,
    "PowerBackup": 9,
    "CarParking": 9,
    "StaffQuarter": 9,
    "Cafeteria": 9,
    "MultipurposeRoom": 9,
    "Hospital": 9,
    "WashingMachine": 9,
    "Gasconnection": 9,
    "AC": 9,
    "Wifi": 9,
    "Childrensplayarea": 9,
    "LiftAvailable": 9,
    "BED": 9,
    "VaastuCompliant": 9,
    "Microwave": 9,
    "GolfCourse": 9,
    "TV": 9,
    "DiningTable": 9,
    "Sofa": 9,
    "Wardrobe": 9,
    "Refrigerator": 9
}
delhi_template = {
    "Price": 6500000,
    "Area": 1000,
    "Location": "Pitampura",
    "No_of_Bedrooms": 2,
    "Resale": 1,
    "MaintenanceStaff": 9,
    "Gymnasium": 9,
    "SwimmingPool": 9,
    "LandscapedGardens": 9,
    "JoggingTrack": 9,
    "RainWaterHarvesting": 9,
    "IndoorGames": 9,
    "ShoppingMall": 9,
    "Intercom": 9,
    "SportsFacility": 9,
    "ATM": 9,
    "ClubHouse": 9,
    "School": 9,
    "24X7Security": 9,
    "PowerBackup": 9,
    "CarParking": 9,
    "StaffQuarter": 9,
    "Cafeteria": 9,
    "MultipurposeRoom": 9,
    "Hospital": 9,
    "WashingMachine": 9,
    "Gasconnection": 9,
    "AC": 9,
    "Wifi": 9,
    "Childrensplayarea": 9,
    "LiftAvailable": 9,
    "BED": 9,
    "VaastuCompliant": 9,
    "Microwave": 9,
    "GolfCourse": 9,
    "TV": 9,
    "DiningTable": 9,
    "Sofa": 9,
    "Wardrobe": 9,
    "Refrigerator": 9
}
chennai_template = {
    "Price": 3700000,
    "Area": 850,
    "Location": "Avadi",
    "No_of_Bedrooms": 2,
    "Resale": 0,
    "MaintenanceStaff": 9,
    "Gymnasium": 9,
    "SwimmingPool": 9,
    "LandscapedGardens": 9,
    "JoggingTrack": 9,
    "RainWaterHarvesting": 9,
    "IndoorGames": 9,
    "ShoppingMall": 9,
    "Intercom": 9,
    "SportsFacility": 9,
    "ATM": 9,
    "ClubHouse": 9,
    "School": 9,
    "24X7Security": 9,
    "PowerBackup": 9,
    "CarParking": 9,
    "StaffQuarter": 9,
    "Cafeteria": 9,
    "MultipurposeRoom": 9,
    "Hospital": 9,
    "WashingMachine": 9,
    "Gasconnection": 9,
    "AC": 9,
    "Wifi": 9,
    "Childrensplayarea": 9,
    "LiftAvailable": 9,
    "BED": 9,
    "VaastuCompliant": 9,
    "Microwave": 9,
    "GolfCourse": 9,
    "TV": 9,
    "DiningTable": 9,
    "Sofa": 9,
    "Wardrobe": 9,
    "Refrigerator": 9
}
hyderabad_template = {
    "Price": 13000000,
    "Area": 1350,
    "Location": "Hitech City",
    "No_of_Bedrooms": 3,
    "Resale": 1,
    "MaintenanceStaff": 0,
    "Gymnasium": 1,
    "SwimmingPool": 1,
    "LandscapedGardens": 0,
    "JoggingTrack": 1,
    "RainWaterHarvesting": 1,
    "IndoorGames": 1,
    "ShoppingMall": 1,
    "Intercom": 1,
    "SportsFacility": 0,
    "ATM": 1,
    "ClubHouse": 1,
    "School": 1,
    "24X7Security": 1,
    "PowerBackup": 1,
    "CarParking": 1,
    "StaffQuarter": 1,
    "Cafeteria": 0,
    "MultipurposeRoom": 0,
    "Hospital": 1,
    "WashingMachine": 0,
    "Gasconnection": 0,
    "AC": 0,
    "Wifi": 1,
    "Childrensplayarea": 1,
    "LiftAvailable": 0,
    "BED": 1,
    "VaastuCompliant": 0,
    "Microwave": 0,
    "GolfCourse": 0,
    "TV": 0,
    "DiningTable": 0,
    "Sofa": 0,
    "Wardrobe": 0,
    "Refrigerator": 0
}
city_templates = {
    "Mumbai": mumbai_template,
    "Delhi": delhi_template,
    "Chennai": chennai_template,
    "Hyderabad": hyderabad_template
}

def predict_price(feature1, feature2, feature3, city):
    default_location_per_city = {
        "Mumbai": "Andheri West",
        "Delhi": "Pitampura",
        "Chennai": "Avadi",
        "Hyderabad": "Hitech City"
    }
    location = default_location_per_city.get(city, "Unknown")

    template = city_templates.get(city)
    if not template:
        return {"error": "City not supported"}

    features_dict = template.copy()
    features_dict["Price"] = feature1
    features_dict["Area"] = feature2
    features_dict["No_of_Bedrooms"] = feature3

    try:
        features_dict["Location_encoded"] = location_le.transform([location])[0]
        features_dict["City_encoded"] = city_le.transform([city])[0]
    except Exception as e:
        return {"error": f"Encoding failed: {e}"}

    features_dict.pop("Location", None)
    features_dict.pop("City", None)

    # Replace the below feature list with exact feature order used in training
    feature_order = [
    "Price",
    "Area",
    "No_of_Bedrooms",
    "Resale",
    "MaintenanceStaff",
    "Gymnasium",
    "SwimmingPool",
    "LandscapedGardens",
    "JoggingTrack",
    "RainWaterHarvesting",
    "IndoorGames",
    "ShoppingMall",
    "Intercom",
    "SportsFacility",
    "ATM",
    "ClubHouse",
    "School",
    "24X7Security",
    "PowerBackup",
    "CarParking",
    "StaffQuarter",
    "Cafeteria",
    "MultipurposeRoom",
    "Hospital",
    "WashingMachine",
    "Gasconnection",
    "AC",
    "Wifi",
    "Childrensplayarea",
    "LiftAvailable",
    "BED",
    "VaastuCompliant",
    "Microwave",
    "GolfCourse",
    "TV",
    "DiningTable",
    "Sofa",
    "Wardrobe",
    "Refrigerator",
    "Location_encoded",
    "City_encoded"
]

    X_values = [features_dict.get(f, 0) for f in feature_order]
    X = np.array([X_values])

    try:
        lgb_pred = lgb_model.predict(X)[0]
        X_scaled = scaler.transform(X)
        nn_pred = nn_model.predict(X_scaled, verbose=0)[0][0]
        ensemble_pred = (lgb_pred + nn_pred) / 2

        return {
            'LightGBM': float(lgb_pred),
            'Deep_Learning': float(nn_pred),
            'Ensemble': float(ensemble_pred),
            'INR_Value': float(ensemble_pred * 100000)
        }
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Prediction failed: {e}\nTraceback:\n{trace}")
        return {"error": f"Prediction failed: {e}"}


# Flask API route handler — note distinct function name to avoid naming collisions
@app.route('/')
def home():
    return "<h2>Flask backend is running! (LightGBM + Deep Learning only)</h2>"

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    default_location_per_city = {
        "Mumbai": "Andheri West",
        "Delhi": "Pitampura",
        "Chennai": "Avadi",
        "Hyderabad": "Hitech City"
    }
    try:
        feature1 = float(data['feature1'])
        feature2 = float(data['feature2'])
        feature3 = float(data['feature3'])
        city = data['city']
        location = default_location_per_city.get(city, "Unknown")
        print(f"Received prediction request: {feature1}, {feature2}, {feature3}, {city}, location used: {location}")
    except Exception as e:
        return jsonify({'error': f'Missing or invalid input: {e}'}), 400

    results = predict_price(feature1, feature2, feature3, city)
    if "error" in results:
        return jsonify(results), 400
    return jsonify(results)

# print("Known location classes in location_le encoder:")
# for loc in location_le.classes_:
#     print(loc)
if __name__ == '__main__':
    app.run(debug=True)
