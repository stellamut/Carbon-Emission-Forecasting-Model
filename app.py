# ===========================================================
# 🌍 SDG 13: Climate Action – Carbon Emissions Forecasting
# Streamlit Dashboard
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# -----------------------------------------------------------
# 🧩 1. App Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="SDG 13: Carbon Emissions Forecasting", layout="wide")

st.title("🌱 SDG 13: Climate Action – Carbon Emissions Forecasting")
st.markdown(
    """
    **Goal:** Use machine learning to forecast global carbon emissions  
    **Approach:** Regression & Clustering | Tools: Python, Scikit-learn, Streamlit  
    """
)

# -----------------------------------------------------------
# 📊 2. Load Dataset
# -----------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasets/co2-fossil-global/master/global.csv"
    df = pd.read_csv(url)
    df.rename(columns={'Year': 'year', 'Total': 'co2_emissions'}, inplace=True)
    df = df.dropna(subset=['year', 'co2_emissions'])
    df = df[df['co2_emissions'] > 0]
    return df

df = load_data()
st.subheader("🌍 Global CO₂ Emissions Data")
st.dataframe(df.head())

# -----------------------------------------------------------
# 🧹 3. Preprocess Data
# -----------------------------------------------------------
df['lag1'] = df['co2_emissions'].shift(1)
df['lag2'] = df['co2_emissions'].shift(2)
df = df.dropna()

X = df[['lag1', 'lag2']]
y = df['co2_emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# 🤖 4. Train Models
# -----------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

results_df = pd.DataFrame(results).T
st.subheader("📈 Model Performance Comparison")
st.dataframe(results_df.style.highlight_min(color="lightcoral", axis=0).highlight_max(color="lightgreen", axis=0))

# -----------------------------------------------------------
# 🔮 5. Forecast Future Emissions
# -----------------------------------------------------------
st.subheader("🔮 Forecast Future CO₂ Emissions")

best_model = models["Random Forest"]
last_two = df[['lag1', 'lag2']].iloc[-1].values.reshape(1, -1)
future_years = st.slider("Select number of years to forecast:", 1, 20, 5)

forecast = []
prev = list(last_two.flatten())
for i in range(future_years):
    scaled_input = scaler.transform([prev[-2:]])
    pred = best_model.predict(scaled_input)[0]
    forecast.append(pred)
    prev.append(pred)

future_df = pd.DataFrame({
    "Year": np.arange(df['year'].max() + 1, df['year'].max() + 1 + future_years),
    "Predicted CO₂ Emissions (kt)": forecast
})

st.write(future_df)

# Plot forecast
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(df['year'], df['co2_emissions'], label="Historical", color='blue')
plt.plot(future_df['Year'], future_df["Predicted CO₂ Emissions (kt)"], label="Forecast", color='red')
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (kt)")
plt.title("Global CO₂ Emissions Forecast")
plt.legend()
st.pyplot(fig)

# -----------------------------------------------------------
# 🔍 6. Optional: Clustering (Unsupervised Learning)
# -----------------------------------------------------------
st.subheader("🔍 Clustering by Emission Levels")

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['co2_emissions']])

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='year', y='co2_emissions', hue='cluster', data=df, palette='viridis')
plt.title("Clustering of Years by Emission Levels")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (kt)")
st.pyplot(fig2)

# -----------------------------------------------------------
# 🌱 7. Ethical Reflection
# -----------------------------------------------------------
st.subheader("🌱 Ethical Reflection")

st.markdown(
    """
    **Bias Considerations:**  
    - Data may be incomplete for developing regions or rely on outdated reporting.  
    - Forecasting based solely on historical emissions ignores socioeconomic or policy interventions.  

    **Fairness & Transparency:**  
    - Open data and interpretable models help promote accountability in climate monitoring.  

    **Sustainability Impact:**  
    - Predictive insights can guide policy and help nations track progress toward SDG 13 targets.  
    """
)

st.success("✅ Dashboard Complete: You’ve built an AI-driven SDG 13 forecasting solution!")

