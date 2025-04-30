import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

# Load model
model = joblib.load("trail_usage_model_daily_with_lags.pkl")

# Proximity dictionary
proximity_miles = {
    '3 Rivers Heritage Trail - Chateau Cyclist': 1.95,
    'Eliza Furnace Trail Cyclist': 2.72,
    'North Shore Trail Cyclist': 417.07,
    'Smithfield St Bridge E Cyclist': 417.07,
    'Smithfield St Bridge W. Cyclist': 417.07,
    'Southside Trail Near 18th St Cyclist': 0.98,
    'Southside Trail near UPMC Cyclist': 0.98,
    'MultiCounter - East Lytle': 1.08,
    'MultiCounter - Hazelwood Trail': 2.93,
    'Multicounter - West Lytle Street': 3.21,
    'Pittsburgh to Millvale': 3.31
}

# Pittsburgh center coordinates
lat, lon = 40.4418, -80.0004

# Use your provided OpenWeatherMap API key
API_KEY = "bcef05a6a7e6e7623f74d7b0ca2b42c0"

# Function to fetch weather forecast
def get_forecast(date_selected):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    target_day = date_selected.strftime('%Y-%m-%d')
    day_forecasts = [item for item in data['list'] if item['dt_txt'].startswith(target_day)]

    if not day_forecasts:
        return None

    temps = [f['main']['temp_max'] for f in day_forecasts]
    wind_speeds = [f['wind']['speed'] for f in day_forecasts]
    rain = [f.get('rain', {}).get('3h', 0) for f in day_forecasts]
    snow = [f.get('snow', {}).get('3h', 0) for f in day_forecasts]

    return {
        'Max Temperature (°C)': max(temps),
        'Precipitation': sum(rain),
        'Snowfall': sum(snow),
        'Wind speed': np.mean(wind_speeds)
    }

# ---------- Streamlit UI ----------
st.title("🚴‍♂️ Trail Usage Predictor (w/ Real Weather)")

trail = st.selectbox("Select a trail:", list(proximity_miles.keys()))
date_input = st.date_input("Choose a date (next 5 days only)", min_value=datetime.today())

lag_1 = st.number_input("Yesterday's usage on this trail", value=50)
rolling_3 = st.number_input("3-day rolling average usage", value=60)

# Get weather
weather = get_forecast(date_input)

if weather:
    st.subheader("📡 Weather Forecast:")
    st.write(weather)

    # Build input
    features = {
        'Max Temperature (°C)': weather['Max Temperature (°C)'],
        'Precipitation': weather['Precipitation'],
        'Snowfall': weather['Snowfall'],
        'Wind speed': weather['Wind speed'],
        'Proximity_miles': proximity_miles[trail],
        'weekday': date_input.weekday(),
        'is_weekend': int(date_input.weekday() in [5,6]),
        'dayofyear': date_input.timetuple().tm_yday,
        'month': date_input.month,
        'lag_1': lag_1,
        'rolling_mean_3': rolling_3,
        f'Trail_{trail}': 1
    }

    # Format for model
    model_input = pd.DataFrame([features])
    for col in model.get_booster().feature_names:
        if col not in model_input.columns:
            model_input[col] = 0
    model_input = model_input[model.get_booster().feature_names]

    if st.button("Predict Usage"):
        pred = model.predict(model_input)[0]
        st.success(f"📈 Predicted trail users for {trail} on {date_input.strftime('%Y-%m-%d')}: **{pred:.0f} users**")
else:
    st.warning("⚠️ Weather data not available for that day. Try a date within the next 5 days.")
