import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration and Initialization ---

# Filepaths for the saved model and scaler (must match paths used in sdg13_carbon_forecast.py)
MODEL_PATH = 'best_model.joblib'
SCALER_PATH = 'scaler.joblib'

# 1. Load Saved Model and Scaler
try:
    # Use st.cache_resource for objects that should only be loaded once
    @st.cache_resource
    def load_assets():
        """Loads the pre-trained model and scaler."""
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler

    model, scaler = load_assets()
    assets_loaded = True
except FileNotFoundError:
    st.error(f"Error: Model or scaler file not found. Ensure '{MODEL_PATH}' and '{SCALER_PATH}' "
             f"are in the same directory as this script and were saved by the training script.")
    assets_loaded = False
except Exception as e:
    st.error(f"An error occurred while loading model assets: {e}")
    assets_loaded = False

# --- UI Setup ---

st.set_page_config(
    page_title="SDG 13: Carbon Emission Forecasting",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header with SDG 13 theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        font-size: 2.5em;
        font-weight: 700;
        color: #1f77b4; /* A deep blue for climate action */
        text-align: center;
        margin-bottom: 20px;
    }
    .sdg-tag {
        font-size: 1.1em;
        font-weight: 600;
        color: #4CAF50; /* Green for sustainability */
        text-align: center;
        margin-bottom: 30px;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #ff7f0e; /* Orange, a color for action/warning */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">AI for Climate Action: Global $\text{CO}_2$ Forecasting</p>', unsafe_allow_html=True)
st.markdown('<p class="sdg-tag">Forecasting Emissions to support SDG 13 – Climate Action</p>', unsafe_allow_html=True)

if assets_loaded:
    # --- Sidebar for Input Parameters (Policy Scenario) ---
    st.sidebar.header("Define Policy Scenario")
    st.sidebar.markdown("Adjust the variables below to simulate future economic and energy policies for a region.")

    # Input Fields
    # The ranges should reflect realistic global data (e.g., Energy_Consumption max ~10,000 to 15,000 TJ for large nations)
    input_year = st.sidebar.slider(
        'Target Year for Forecast',
        min_value=2025, max_value=2050, value=2030, step=1,
        help="The year for which the emission forecast is needed."
    )

    input_gdp = st.sidebar.number_input(
        'Projected GDP (Billion USD)',
        min_value=0.1, max_value=30000.0, value=1500.0, step=50.0,
        help="Hypothetical GDP for the target year (in billions of USD)."
    )

    input_population = st.sidebar.number_input(
        'Projected Population (Millions)',
        min_value=0.5, max_value=2000.0, value=350.0, step=10.0,
        help="Estimated population for the target year (in millions of people)."
    )

    input_energy = st.sidebar.number_input(
        'Projected Energy Consumption (TJ)',
        min_value=10.0, max_value=20000.0, value=4500.0, step=100.0,
        help="Total energy consumption projected for the target year (in Terajoules)."
    )

    # Prediction Button
    predict_button = st.sidebar.button("Forecast Emissions", use_container_width=True)

    # --- Main Content Area ---

    # Display the current scenario being simulated
    st.subheader("Current Scenario Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Year", input_year)
    col2.metric("Projected GDP", f"{input_gdp:,.0f} Billion USD")
    col3.metric("Projected Population", f"{input_population:,.0f} Million")
    col4.metric("Projected Energy Use", f"{input_energy:,.0f} TJ")

    st.markdown("---")

    # --- Prediction Logic ---
    if predict_button:
        with st.spinner("Calculating $\text{CO}_2$ Forecast based on scenario..."):
            # 1. Prepare Input Data Frame (Must match the training feature order)
            input_df = pd.DataFrame({
                'Energy_Consumption_TJ': [input_energy],
                'GDP_Billion': [input_gdp],
                'Population_Million': [input_population],
                'Year': [input_year]
            })

            # 2. Scale the input data using the loaded scaler
            # The scaler transforms data based on the mean/std DEV of the training data
            scaled_input = scaler.transform(input_df)

            # 3. Make Prediction
            prediction = model.predict(scaled_input)
            predicted_emissions = prediction[0]

            # 4. Display Results
            st.success("✅ Forecast Complete!")

            st.markdown("## Predicted $\text{CO}_2$ Emissions")
            
            # Using a custom container for the large result
            st.markdown(f"""
                <div style='background-color: #e0f7fa; padding: 20px; border-radius: 10px; border: 2px solid #00bcd4; text-align: center;'>
                    <p style='font-size: 1.2em; color: #00838f; margin: 0;'>Projected $\text{{CO}}_2$ Emissions (KT)</p>
                    <h1 style='font-size: 4em; color: #004d40; margin: 5px 0 0;'>
                        {predicted_emissions:,.0f}
                    </h1>
                    <p style='font-size: 0.9em; color: #333;'>KT = Kilo tonnes (metric tons) of $\text{{CO}}_2$ equivalent.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")

            # --- Interpretation and Policy Context ---
            st.subheader("Policy Implications")
            st.info(f"For the year **{input_year}** and the defined economic scenario, the model forecasts a total emission of **{predicted_emissions:,.0f} KT**.")
            
            st.markdown("""
            * **Scenario Evaluation:** Use this tool to test different policy outcomes. For instance, lowering the **Projected Energy Consumption (TJ)** while maintaining or increasing **GDP** simulates an energy efficiency or renewable energy transition policy.
            * **Model Reliability:** This forecast is based on historical relationships between economic activity (GDP), energy use, and population. Continuous data input and re-training are necessary for maximum accuracy.
            """)

    # --- Footer Information ---
    st.markdown("---")
    st.caption("Model built for SDG 13 - Climate Action. Developed using Scikit-learn and Streamlit.")
