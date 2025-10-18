import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
# Streamlit page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Carbon Emission Forecast (SDG 13)",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for theming based on the provided image and Futura font
st.markdown("""
    <style>
    /* Google Fonts for Futura-like sans-serif */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap'); /* Montserrat is a good Futura alternative */

    html, body, [class*="st-"] {
        font-family: 'Montserrat', sans-serif;
    }

    /* General background - light cream from the image */
    .stApp {
        background-color: #fcfaf5; /* A light, creamy background */
    }

    /* Sidebar background - could be a darker tone or the dark green */
    .css-vk329t, .css-1dp5x4i { /* Specific Streamlit sidebar classes */
        background-color: #1a4a35; /* Dark green from the image */
        color: #fcfaf5; /* Creamy text for sidebar */
    }
    .css-1dp5x4i .css-1oe5f06, .css-vk329t .css-1oe5f06 { /* Sidebar header/title */
        color: #fcfaf5;
    }
    .css-1dp5x4i .css-1oe5f06 a, .css-vk329t .css-1oe5f06 a { /* Links in sidebar */
        color: #a8dadc; /* A soft blue-green for links */
    }

    /* Main content text color */
    .main .block-container {
        color: #333333; /* Darker text for readability */
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a4a35; /* Dark green for headings */
    }

    /* Buttons - Golden Yellow for primary action */
    .stButton > button {
        background-color: #ffb800; /* Golden yellow */
        color: white; /* White text on yellow */
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e6a700; /* Slightly darker yellow on hover */
        color: white;
    }

    /* Sliders - use light green/peach accents */
    .stSlider > div > div > div[data-baseweb="slider"] > div:nth-child(1) {
        background-color: #8bc34a; /* Light green track */
    }
    .stSlider > div > div > div[data-baseweb="slider"] > div:nth-child(2) {
        background-color: #ffb800; /* Golden yellow thumb */
    }

    /* Expander/Container backgrounds - use lighter green or peach */
    .streamlit-expanderHeader {
        background-color: #8bc34a; /* Light green for expander headers */
        color: white !important;
        border-radius: 5px;
    }
    .streamlit-expanderContent {
        background-color: #f5f5f5; /* Light grey for content */
        border-left: 3px solid #8bc34a;
        padding: 15px;
        border-radius: 0 0 5px 5px;
    }

    /* Info/Success/Warning/Error boxes */
    .st-emotion-cache-16idsj4 p { /* text in success/info etc. boxes */
        font-size: 1rem;
    }
    .st-emotion-cache-1f8rbrq { /* Streamlit info box base */
        background-color: #e0f2f7 !important; /* light blue */
        color: #007bff !important;
        border-left: 5px solid #007bff !important;
    }
    .st-emotion-cache-1k0f07p { /* Streamlit success box base */
        background-color: #e8f5e9 !important; /* light green */
        color: #28a745 !important;
        border-left: 5px solid #28a745 !important;
    }
    /* You can define custom success/info based on your palette */
    /* Example: using peach for success/info messages */
    .stAlert.info {
        background-color: #ffe0b2; /* Peach */
        color: #e65100; /* Darker orange for text */
        border-left: 5px solid #e65100;
    }
    .stAlert.success {
        background-color: #c8e6c9; /* Lighter light green */
        color: #2e7d32; /* Darker green */
        border-left: 5px solid #2e7d32;
    }

    /* Metric elements (like st.metric) */
    [data-testid="stMetricValue"] {
        color: #1a4a35 !important; /* Dark green for metric value */
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #e65100 !important; /* Peach for delta */
    }


    </style>
""", unsafe_allow_html=True)

# --- Asset Paths ---
# Ensure these paths match where your model and scaler are saved!
# Check your `sdg13_carbon_forecast.py` output for the exact model name.
MODEL_PATH = 'random_forest_regressor_model.joblib' # Assuming Random Forest was best
SCALER_PATH = 'scaler.joblib'

# --- Load Model and Scaler ---
@st.cache_resource # Cache the model and scaler to avoid reloading on every rerun
def load_model_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' are in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model assets: {e}")
        st.stop()

model, scaler = load_model_assets()

# --- Feature Definitions ---
FEATURES = ['GDP_Billion', 'Energy_Consumption_TJ', 'Population_Million', 'Year']

# --- Title and Introduction ---
st.title("🌍 Carbon Emission Forecast (SDG 13: Climate Action)")
st.markdown("""
    Welcome to the AI-powered Carbon Emission Forecasting tool. This application leverages a trained Machine Learning model to predict future CO₂ emissions based on key economic, energy, and demographic indicators.
    
    Use the sidebar to input your desired values and see the forecasted carbon footprint. This tool aims to assist policymakers in evaluating mitigation strategies and promoting sustainable practices for a greener future.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Adjust the sliders to set the values for forecasting CO₂ emissions.")

# Input fields for features
gdp = st.sidebar.slider("GDP (Billion USD)", min_value=100.0, max_value=10000.0, value=2500.0, step=50.0)
energy_consumption = st.sidebar.slider("Energy Consumption (TeraJoules)", min_value=50.0, max_value=2000.0, value=500.0, step=10.0)
population = st.sidebar.slider("Population (Million)", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
year = st.sidebar.slider("Year of Forecast", min_value=2020, max_value=2050, value=2030, step=1)

# --- Prediction Button ---
if st.sidebar.button("Predict CO₂ Emissions"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[gdp, energy_consumption, population, year]], columns=FEATURES)

    # Scale the input data using the loaded scaler
    # Note: We use .transform(), not .fit_transform() as the scaler is already fitted
    scaled_input_data = pd.DataFrame(scaler.transform(input_data), columns=FEATURES)

    # Make prediction
    prediction = model.predict(scaled_input_data)[0]

    # --- Display Results ---
    st.header("Forecast Results")
    st.success(f"### Predicted CO₂ Emissions: **{prediction:,.2f} KT**")

    st.markdown("""
        <div style="background-color:#ffe0b2; padding:15px; border-radius:10px; border-left: 5px solid #e65100; margin-top:20px;">
            <h4 style="color:#e65100; margin-top:0;">Policy Insight</h4>
            <p style="color:#e65100;">
            Based on the current model, this forecast highlights potential emission levels. For effective climate action, consider focusing on strategies that:</p>
            <ul>
                <li style="color:#e65100;">Promote **renewable energy** sources to reduce reliance on fossil fuels.</li>
                <li style="color:#e65100;">Implement **energy efficiency** measures across industries and residential sectors.</li>
                <li style="color:#e65100;">Invest in **sustainable economic growth** models that decouple GDP growth from carbon intensity.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Optional: Display feature importance if using Random Forest
    if isinstance(model, joblib.ParallelBackendBase) or isinstance(model, type(joblib.load(MODEL_PATH))): # Check if it's a RandomForest model for feature_importances
        if hasattr(model, 'feature_importances_'):
            st.subheader("Key Factors Influencing Emissions")
            feature_importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
            
            st.info("The chart below illustrates which factors had the most significant impact on the emission forecast according to our model.")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.plot(kind='barh', ax=ax, color='#ffb800') # Golden yellow bars
            ax.set_title("Feature Importance", color='#1a4a35') # Dark green title
            ax.set_xlabel("Importance", color='#333333')
            ax.set_ylabel("Feature", color='#333333')
            plt.tight_layout()
            st.pyplot(fig)

# --- About Section (Expander) ---
with st.expander("About This Project"):
    st.markdown("""
        This project aligns with **UN Sustainable Development Goal 13: Climate Action**.
        Our objective is to provide a data-driven tool for understanding and forecasting CO₂ emissions,
        empowering policymakers with insights to drive global efforts in reducing carbon footprints
        and mitigating climate change impacts.

        The model uses historical data of GDP, Energy Consumption, and Population to predict
        CO₂ emissions using a Random Forest Regressor algorithm.
        
        **Built with ❤️ for a Sustainable Future.**
    """)

# --- Footer ---
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #888888; font-size: 0.8em;">
        Version 1.0 | Developed for SDG 13 - Climate Action
    </div>
""", unsafe_allow_html=True)
