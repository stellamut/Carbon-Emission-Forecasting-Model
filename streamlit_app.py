import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib # Added for model persistence (deployment readiness)

# --- STRETCH GOAL 1: Real-Time Data Integration Placeholder ---
def fetch_real_time_data(country_name, current_year):
    """
    Conceptual function for integrating real-time data via APIs.
    
    In a real application, this function would:
    1. Make API calls (e.g., to a global weather service, economic data source).
    2. Process the JSON/XML response into a format ready for prediction.
    3. Handle API keys, rate limits, and error handling (exponential backoff).
    
    Returns a dictionary of current feature values for a single prediction.
    """
    print(f"\n[Conceptual API Call] Fetching data for {country_name}, {current_year}...")
    # Simulate fresh, slightly varying data for a single country/year
    return {
        'GDP_Billion': 4500.0 + np.random.normal(0, 50),
        'Energy_Consumption_TJ': 950.0 + np.random.normal(0, 20),
        'Population_Million': 105.0 + np.random.normal(0, 1),
        'Year': current_year
    }

# --- 1. Synthetic Data Generation (Mimicking World Bank/UN Data) ---
# Since we cannot access live external data, we simulate a dataset
# for 10 years and 100 hypothetical 'Countries/Regions'.

def create_synthetic_data(num_entries=1000):
    """
    Generates synthetic data for carbon emission forecasting.
    Features mimic real-world drivers: GDP, Energy Consumption, Population.
    """
    np.random.seed(42)

    # 100 countries over 10 years (1000 data points)
    countries = [f'Country_{i+1}' for i in range(100)]
    years = np.arange(2010, 2020)
    
    # Create a base dataframe
    data = []
    for _ in range(num_entries):
        country = np.random.choice(countries)
        year = np.random.choice(years)
        
        # Base trends (larger countries/later years generally have higher values)
        base_gdp = np.random.uniform(500, 5000) * (1 + (year - 2010) * 0.05)
        base_energy = np.random.uniform(100, 1000) * (1 + (year - 2010) * 0.03)
        base_pop = np.random.uniform(10, 100) * (1 + (year - 2010) * 0.01)
        
        # Add country-specific noise/profile (simulating industrial vs. service economies)
        country_factor = countries.index(country) / 100.0
        
        gdp = base_gdp * (1 + country_factor * 0.5) + np.random.normal(0, 50)
        energy_consumption = base_energy * (1 + country_factor * 0.3) + np.random.normal(0, 30)
        population = base_pop + np.random.normal(0, 5)

        # Target Variable (CO2 Emissions - Kilo tonnes)
        # Emissions = (High Weight on Energy * Factor) + (Medium Weight on GDP) + Noise
        co2_emissions = (energy_consumption * 1.5) + (gdp * 0.1) + (population * 0.5) + np.random.normal(0, 10)
        
        data.append({
            'Country': country,
            'Year': year,
            'GDP_Billion': gdp,
            'Energy_Consumption_TJ': energy_consumption,
            'Population_Million': population,
            'CO2_Emissions_KT': co2_emissions
        })

    df = pd.DataFrame(data)
    
    # Introduce a few missing values for the preprocessing step demonstration
    df.loc[df.sample(frac=0.01).index, 'Energy_Consumption_TJ'] = np.nan
    df.loc[df.sample(frac=0.005).index, 'CO2_Emissions_KT'] = np.nan
    
    return df

# --- 2. ML Pipeline Implementation ---

def run_forecasting_model():
    """
    Executes the full machine learning pipeline for CO2 emission forecasting.
    """
    print("------------------------------------------------------------------")
    print("🎯 SDG 13: AI for Climate Action - Carbon Emission Forecasting")
    print("------------------------------------------------------------------\n")

    # --- Data Generation & Initial Look ---
    print("1. Generating Synthetic Dataset (Simulating World Bank/UN Data)...")
    df = create_synthetic_data()
    
    # --- Data Preprocessing ---
    print("\n2. Data Preprocessing...")
    
    # a) Handle Missing Values: Drop rows where the target variable is missing
    df.dropna(subset=['CO2_Emissions_KT'], inplace=True)
    
    # b) Handle Missing Features: Impute feature NaNs with the mean
    # FIX: Replaced inplace=True with direct assignment to prevent Pandas FutureWarning
    df['Energy_Consumption_TJ'] = df['Energy_Consumption_TJ'].fillna(df['Energy_Consumption_TJ'].mean())
    
    # Define Features (X) and Target (y)
    features = ['GDP_Billion', 'Energy_Consumption_TJ', 'Population_Million', 'Year']
    X = df[features]
    y = df['CO2_Emissions_KT']
    
    # c) Feature Scaling (Normalization) - Critical for consistent deployment
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=features)

    # d) Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Model Training and Comparison (STRETCH GOAL: Compare Multiple Algorithms) ---
    print("\n3. Model Training and Evaluation (Comparing 3 Models)...")
    
    # Define models, including Ridge Regression for better Linear performance
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0), # Ridge is often better than pure Linear Reg.
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    }

    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # --- Evaluation ---
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'y_pred': y_pred,
            'model': model # Store the trained model instance
        }
        
        print(f"MAE (Mean Absolute Error): {mae:.2f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
        print(f"R² (Coefficient of Determination): {r2:.4f}")

    # --- Final Outcomes & Deployment Preparation ---
    print("\n4. Final Outcomes and Deployment Preparation...")
    
    # Select Best Model based on R2 score (higher is better)
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_r2 = results[best_model_name]['R2']
    best_model = results[best_model_name]['model']
    
    print(f"\nOptimal Model Selected: {best_model_name} (R² = {best_r2:.4f})")
    
    # STRETCH GOAL 2: Deploy Model (Save model and scaler objects)
    # The saved model and scaler can be loaded by a Flask/Streamlit app.
    model_filepath = f'{best_model_name.replace(" ", "_").lower()}_model.joblib'
    scaler_filepath = 'scaler.joblib'
    
    joblib.dump(best_model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"\nModel Persistence (Deployment Readiness):")
    print(f"-> Trained model saved to: {model_filepath}")
    print(f"-> Scaler saved to: {scaler_filepath} (ESSENTIAL for new data preprocessing)")
    print("This allows a Streamlit/Flask app to load the model without retraining.")
    
    # Simulate a visualization showing Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, results[best_model_name]['y_pred'], alpha=0.6, color='darkred')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title(f'Actual vs. Predicted CO₂ Emissions ({best_model_name})', fontsize=14)
    plt.xlabel('Actual CO₂ Emissions (KT)', fontsize=12)
    plt.ylabel('Predicted CO₂ Emissions (KT)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Demonstration of Real-Time Prediction (combining STRETCH GOAL 1 & 2)
    print("\nDemonstration of Real-Time Prediction using the saved components:")
    
    # 1. Fetch 'real-time' data (STRETCH GOAL 1)
    new_data = fetch_real_time_data("Global Region X", 2025)
    new_df = pd.DataFrame([new_data], columns=features)
    
    # 2. Preprocess data using the SAVED scaler
    # Note: We use scaler.transform(), NOT scaler.fit_transform()
    # FIX: Convert scaled NumPy array back to DataFrame to prevent Scikit-learn UserWarning
    new_data_scaled_df = pd.DataFrame(scaler.transform(new_df), columns=features)
    
    # 3. Predict using the SAVED model
    predicted_emission = best_model.predict(new_data_scaled_df)[0]
    
    print(f"\nPolicy Recommendation and Forecast:")
    print(f"-> Forecasted CO₂ Emission for 2025 (Region X): {predicted_emission:.2f} KT")
    
    if best_model_name == "Random Forest Regressor":
        # Feature Importance is a key benefit of Random Forests for policy makers
        feature_importances = pd.Series(
            best_model.feature_importances_, 
            index=features
        ).sort_values(ascending=False)
        
        print("\nTop 3 Feature Importance (for Policy Prioritization):")
        for i, (feature, importance) in enumerate(feature_importances.head(3).items()):
            print(f"  {i+1}. {feature}: {importance:.4f}")
            
        print("\nPolicy Insight:")
        print("-> The model suggests that Energy Consumption is the most critical driver of CO2 emissions.")
        print("-> Policymakers should prioritize investments in renewable energy infrastructure and industrial efficiency upgrades to achieve the highest impact on emission reduction (SDG 13).")
    else:
        print("\nPolicy Insight:")
        print("-> The linear nature of the model suggests proportional impacts.")
        print("-> Focus on systemic economic shifts (GDP decoupling) and population planning alongside energy transition for mitigation.")


if __name__ == "__main__":
    run_forecasting_model()

