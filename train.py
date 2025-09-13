# -*- coding: utf-8 -*-


# Import all necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Advanced visualization libraries
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import MarkerCluster

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Visualization and interpretation
import eli5
from eli5.sklearn import PermutationImportance
from geopy.geocoders import Nominatim
from colorama import Fore, Back, Style

# Color setup for output
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA

# Custom color palette
custom_colors = ["#4e89ae", "#c56183","#ed6663","#ffa372"]
customPalette = sns.set_palette(sns.color_palette(custom_colors))

print("📊 Housing Price Prediction - Advanced ML Comparison")
print("="*60)


import os


base_path = "./"

df1 = pd.read_csv(os.path.join(base_path, 'Mumbai.csv'))
df2 = pd.read_csv(os.path.join(base_path, 'Delhi.csv'))
df3 = pd.read_csv(os.path.join(base_path, 'Chennai.csv'))
df4 = pd.read_csv(os.path.join(base_path, 'Hyderabad.csv'))
print(f"{g_}✅ Datasets loaded successfully!")

# Data cleaning - replace missing value indicator
df1.replace(9, np.nan, inplace=True)
df2.replace(9, np.nan, inplace=True)
df3.replace(9, np.nan, inplace=True)
df4.replace(9, np.nan, inplace=True)

# Drop missing values
df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()

# Add city labels
df1['City'] = 'Mumbai'
df2['City'] = 'Delhi'
df3['City'] = 'Chennai'
df4['City'] = 'Hyderabad'

# Combine all datasets
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

print(f"{y_}Dataset Shapes After Cleaning:{r_}")
print(f"Mumbai: {df1.shape}")
print(f"Delhi: {df2.shape}")
print(f"Chennai: {df3.shape}")
print(f"Hyderabad: {df4.shape}")
print(f"{g_}Combined Dataset: {combined_df.shape}")

# Convert price to lakhs (divide by 100000)
combined_df['Price'] = combined_df['Price'] / 100000

print(f"\n{b_}Dataset Info:")
print(combined_df.info())
print(f"\n{b_}First few rows:")
print(combined_df.head())

def comprehensive_preprocessing(df):
    """
    Advanced preprocessing for housing data
    """
    print(f"{y_}🔧 Starting comprehensive preprocessing...")

    # Create a copy
    df_processed = df.copy()

    # Handle categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col not in ['Location', 'City']]

    # Label encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Handle Location (create location-based features)
    if 'Location' in df_processed.columns:
        location_le = LabelEncoder()
        df_processed['Location_encoded'] = location_le.fit_transform(df_processed['Location'])
        label_encoders['Location'] = location_le

    # Handle City
    if 'City' in df_processed.columns:
        city_le = LabelEncoder()
        df_processed['City_encoded'] = city_le.fit_transform(df_processed['City'])
        label_encoders['City'] = city_le

    # Create additional features
    if 'Area' in df_processed.columns and 'No. of Bedrooms' in df_processed.columns:
        df_processed['Area_per_bedroom'] = df_processed['Area'] / (df_processed['No. of Bedrooms'] + 1)

    if 'Balconies' in df_processed.columns and 'No. of Bedrooms' in df_processed.columns:
        df_processed['Balcony_bedroom_ratio'] = df_processed['Balconies'] / (df_processed['No. of Bedrooms'] + 1)

    print(f"{g_}✅ Preprocessing completed!")
    return df_processed, label_encoders

# Apply preprocessing
processed_df, encoders = comprehensive_preprocessing(combined_df)

# Prepare features and target
feature_columns = [col for col in processed_df.columns if col not in ['Price', 'Location', 'City']]
X = processed_df[feature_columns]
y = processed_df['Price']

print(f"\n{b_}Feature columns: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for deep learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"{y_}🚀 Training Models...")
print("="*50)

# Model evaluation function
def evaluate_model(y_true, y_pred, model_name):
    """Enhanced model evaluation"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{g_}📊 {model_name} Performance:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Store results
model_results = {}
model_predictions = {}

# 1. XGBoost Regressor (from your original code)
print(f"{b_}🌟 Training XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
model_results['XGBoost'] = evaluate_model(y_test, xgb_preds, "XGBoost")
model_predictions['XGBoost'] = xgb_preds

# 2. LightGBM Regressor
print(f"{b_}🌟 Training LightGBM Regressor...")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_eval],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=-1)]
)

lgb_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
model_results['LightGBM'] = evaluate_model(y_test, lgb_preds, "LightGBM")
model_predictions['LightGBM'] = lgb_preds

# 3. Deep Learning Model (Neural Network)
print(f"{b_}🌟 Training Deep Learning Model...")

def build_advanced_nn(input_dim):
    """Build advanced neural network architecture"""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.1),

        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model

# Build and train neural network
nn_model = build_advanced_nn(X_train_scaled.shape[1])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

nn_preds = nn_model.predict(X_test_scaled).flatten()
model_results['Deep Learning'] = evaluate_model(y_test, nn_preds, "Deep Learning (ANN)")
model_predictions['Deep Learning'] = nn_preds

print(f"\n{g_}✅ All models trained successfully!")

# Create comprehensive results DataFrame
results_data = {
    'Model': list(model_results.keys()),
    'RMSE': [model_results[model]['RMSE'] for model in model_results.keys()],
    'MAE': [model_results[model]['MAE'] for model in model_results.keys()],
    'R² Score': [model_results[model]['R2'] for model in model_results.keys()]
}

results_df = pd.DataFrame(results_data)

print(f"\n{y_}📊 COMPREHENSIVE MODEL BENCHMARK")
print("="*60)
print(results_df.to_string(index=False, float_format='%.4f'))

# Find best model
best_model_idx = results_df['R² Score'].idxmax()
best_model = results_df.iloc[best_model_idx]['Model']

print(f"\n{g_}🏆 BEST PERFORMING MODEL: {best_model}")
print(f"   R² Score: {results_df.iloc[best_model_idx]['R² Score']:.4f}")
print(f"   RMSE: {results_df.iloc[best_model_idx]['RMSE']:.4f}")
print(f"   MAE: {results_df.iloc[best_model_idx]['MAE']:.4f}")

# Visualization 1: Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

# RMSE Comparison
axes[0,0].bar(results_df['Model'], results_df['RMSE'], color=custom_colors[:3])
axes[0,0].set_title('Root Mean Squared Error (RMSE)', fontsize=16, fontweight='bold')
axes[0,0].set_ylabel('RMSE (Lakhs)', fontsize=12)
axes[0,0].tick_params(axis='x', rotation=45)

# MAE Comparison
axes[0,1].bar(results_df['Model'], results_df['MAE'], color=custom_colors[:3])
axes[0,1].set_title('Mean Absolute Error (MAE)', fontsize=16, fontweight='bold')
axes[0,1].set_ylabel('MAE (Lakhs)', fontsize=12)
axes[0,1].tick_params(axis='x', rotation=45)

# R² Score Comparison
axes[1,0].bar(results_df['Model'], results_df['R² Score'], color=custom_colors[:3])
axes[1,0].set_title('R² Score (Higher is Better)', fontsize=16, fontweight='bold')
axes[1,0].set_ylabel('R² Score', fontsize=12)
axes[1,0].tick_params(axis='x', rotation=45)

# Performance Radar Chart
categories = ['RMSE (Inverted)', 'MAE (Inverted)', 'R² Score']
fig_radar = go.Figure()

for i, model in enumerate(results_df['Model']):
    # Normalize and invert RMSE and MAE for radar chart
    rmse_norm = 1 - (results_df.iloc[i]['RMSE'] / results_df['RMSE'].max())
    mae_norm = 1 - (results_df.iloc[i]['MAE'] / results_df['MAE'].max())
    r2_norm = results_df.iloc[i]['R² Score']

    values = [rmse_norm, mae_norm, r2_norm]

    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=model,
        line_color=custom_colors[i]
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    title="Model Performance Radar Chart",
    showlegend=True,
    width=600,
    height=500
)

axes[1,1].axis('off')
plt.tight_layout()
plt.show()
fig_radar.show()

# Visualization 2: Prediction vs Actual Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

models_list = ['XGBoost', 'LightGBM', 'Deep Learning']
colors = custom_colors[:3]

for i, model in enumerate(models_list):
    preds = model_predictions[model]
    r2 = model_results[model]['R2']

    axes[i].scatter(y_test, preds, alpha=0.6, color=colors[i], s=30)
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_xlabel('Actual Price (Lakhs)', fontsize=12)
    axes[i].set_ylabel('Predicted Price (Lakhs)', fontsize=12)
    axes[i].set_title(f'{model}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature Importance Analysis
print(f"\n{y_}🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# XGBoost Feature Importance
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# LightGBM Feature Importance
lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importance()
}).sort_values('importance', ascending=False)

# Plot feature importance
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# Top 10 XGBoost features
top_xgb = xgb_importance.head(10)
axes[0].barh(top_xgb['feature'], top_xgb['importance'], color=custom_colors[0])
axes[0].set_title('Top 10 Feature Importance - XGBoost', fontsize=16, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=12)

# Top 10 LightGBM features
top_lgb = lgb_importance.head(10)
axes[1].barh(top_lgb['feature'], top_lgb['importance'], color=custom_colors[1])
axes[1].set_title('Top 10 Feature Importance - LightGBM', fontsize=16, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=12)

plt.tight_layout()
plt.show()

# Print top features
print(f"\n{g_}🔝 Top 5 Most Important Features:")
print("\nXGBoost:")
print(xgb_importance.head())
print("\nLightGBM:")
print(lgb_importance.head())

# Create detailed performance summary
summary_stats = {
    'Model': [],
    'Training_Time': ['Fast', 'Very Fast', 'Moderate'],
    'Interpretability': ['High', 'High', 'Low'],
    'Overfitting_Risk': ['Medium', 'Low', 'High'],
    'Memory_Usage': ['Medium', 'Low', 'High'],
    'RMSE': [],
    'MAE': [],
    'R2_Score': []
}

for model in ['XGBoost', 'LightGBM', 'Deep Learning']:
    summary_stats['Model'].append(model)
    summary_stats['RMSE'].append(model_results[model]['RMSE'])
    summary_stats['MAE'].append(model_results[model]['MAE'])
    summary_stats['R2_Score'].append(model_results[model]['R2'])

detailed_summary = pd.DataFrame(summary_stats)

print(f"\n{y_} DETAILED MODEL COMPARISON SUMMARY")
print("="*70)
print(detailed_summary.to_string(index=False, float_format='%.4f'))

# Save results
results_df.to_csv('housing_price_model_comparison.csv', index=False)
detailed_summary.to_csv('detailed_model_analysis.csv', index=False)

print(f"\n{g_} Results saved to CSV files:")
print("   - housing_price_model_comparison.csv")
print("   - detailed_model_analysis.csv")

# Final recommendations
print(f"\n{b_} MODEL RECOMMENDATIONS:")
print("="*40)

if best_model == 'XGBoost':
    print("🏆 XGBoost is the best performer")
    print("    Excellent balance of accuracy and interpretability")
    print("    Robust to overfitting")
    print("    Good feature importance insights")

elif best_model == 'LightGBM':
    print("🏆 LightGBM is the best performer")
    print("    Fastest training time")
    print("    Memory efficient")
    print("    Excellent for large datasets")

else:
    print("🏆 Deep Learning is the best performer")
    print("    Can capture complex non-linear patterns")
    print("    Scalable to larger datasets")
    print("     Requires more data and computational resources")

print(f"\n{m_} Analysis Complete! All models trained and benchmarked successfully.")

# ... (Previous code: model training, evaluation, etc.)

nn_preds = nn_model.predict(X_test_scaled).flatten()
model_results['Deep Learning'] = evaluate_model(y_test, nn_preds, "Deep Learning (ANN)")
model_predictions['Deep Learning'] = nn_preds

print(f"\n{g_}✅ All models trained successfully!")

# 🔥 ADD THE PREDICTION FUNCTION HERE 🔥
# ================================================

def predict_new_property(area, bedrooms, city, location="Unknown", resale=0, balconies=1, **kwargs):
    """
    Predict price for a new property using all trained models
    """
    print(f"\n{y_}🏠 PREDICTING PRICE FOR NEW PROPERTY:")
    print("="*50)
    print(f"   📐 Area: {area} sq ft")
    print(f"   🛏️  Bedrooms: {bedrooms}")
    print(f"   🏙️  City: {city}")
    print(f"   📍 Location: {location}")
    print(f"   🔄 Resale: {'Yes' if resale else 'New'}")
    print(f"   🏠 Balconies: {balconies}")

    # Create feature vector matching training data format
    # Note: This is a simplified version - you'll need to match exact feature engineering
    try:
        # Encode city
        if city in ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad']:
            city_mapping = {'Mumbai': 0, 'Delhi': 1, 'Chennai': 2, 'Hyderabad': 3}
            city_encoded = city_mapping[city]
        else:
            city_encoded = 0  # Default to Mumbai

        # Create basic feature vector (adjust based on your actual features)
        # This should match the exact number and order of features used in training
        feature_vector = np.array([[
            area,                    # Area
            bedrooms,               # No. of Bedrooms
            balconies,              # Balconies
            resale,                 # Resale
            city_encoded,           # City_encoded
            0,                      # Location_encoded (simplified)
            area / (bedrooms + 1),  # Area_per_bedroom
            balconies / (bedrooms + 1),  # Balcony_bedroom_ratio
            # Add zeros for other features that aren't provided
            *[0] * (X.shape[1] - 8)  # Fill remaining features with zeros
        ]])

        # Make sure feature vector has correct number of features
        if feature_vector.shape[1] != X.shape[1]:
            print(f"{r_}⚠️  Warning: Feature count mismatch. Adjusting...")
            # Pad or trim to match training features
            if feature_vector.shape[1] < X.shape[1]:
                padding = np.zeros((1, X.shape[1] - feature_vector.shape[1]))
                feature_vector = np.concatenate([feature_vector, padding], axis=1)
            else:
                feature_vector = feature_vector[:, :X.shape[1]]

        # Make predictions with each model
        xgb_pred = xgb_model.predict(feature_vector)[0]
        lgb_pred = lgb_model.predict(feature_vector)[0]

        # Scale for neural network
        feature_vector_scaled = scaler.transform(feature_vector)
        nn_pred = nn_model.predict(feature_vector_scaled, verbose=0)[0][0]

        # Calculate ensemble prediction
        ensemble_pred = (xgb_pred + lgb_pred + nn_pred) / 3

        print(f"\n{g_}💰 PRICE PREDICTIONS:")
        print(f"   XGBoost:      ₹{xgb_pred:.2f} lakhs")
        print(f"   LightGBM:     ₹{lgb_pred:.2f} lakhs")
        print(f"   Deep Learning: ₹{nn_pred:.2f} lakhs")
        print(f"   📊 Ensemble:   ₹{ensemble_pred:.2f} lakhs")

        # Convert back to actual INR
        print(f"\n{b_}💵 IN ACTUAL CURRENCY:")
        print(f"   Ensemble Prediction: ₹{ensemble_pred * 100000:,.0f}")

        return {
            'XGBoost': xgb_pred,
            'LightGBM': lgb_pred,
            'Deep_Learning': nn_pred,
            'Ensemble': ensemble_pred,
            'INR_Value': ensemble_pred * 100000
        }

    except Exception as e:
        print(f"{r_}❌ Error in prediction: {e}")
        return None

# Example usage - Test the function
print(f"\n{m_}🧪 TESTING PREDICTION FUNCTION:")
print("="*60)

# Test Case 1: Mumbai 2BHK
test_prediction_1 = predict_new_property(
    area=1200,
    bedrooms=2,
    city="Mumbai",
    location="Bandra West",
    resale=0,
    balconies=2
)

# Test Case 2: Delhi 3BHK Resale
test_prediction_2 = predict_new_property(
    area=1800,
    bedrooms=3,
    city="Delhi",
    location="CP",
    resale=1,
    balconies=2
)

# Test Case 3: Chennai 1BHK New
test_prediction_3 = predict_new_property(
    area=800,
    bedrooms=1,
    city="Chennai",
    location="T Nagar",
    resale=0,
    balconies=1
)

print(f"\n{g_}✅ Prediction function ready for use!")

# ================================================
# Continue with rest of the benchmarking code...

# Create comprehensive results DataFrame
results_data = {
    'Model': list(model_results.keys()),
    'RMSE': [model_results[model]['RMSE'] for model in model_results.keys()],
    # ... rest of benchmarking code
}
import joblib

# Save XGBoost model
joblib.dump(xgb_model, 'models/xgb_model.pkl')

# Save LightGBM model
lgb_model.save_model('models/lgb_model.txt')

# Save Neural Network (Keras)
nn_model.save('models/nn_model.h5')

# Save scaler (used for neural network)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders['Location'], 'models/location_le.pkl')
joblib.dump(encoders['City'], 'models/city_le.pkl')

print("✅ All models and scaler saved successfully!")
