import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ------------------- MODEL & CONFIG ---------------------
model = joblib.load("trail_usage_model_daily_with_lags.pkl")

API_KEY = "bcef05a6a7e6e7623f74d7b0ca2b42c0"
lat, lon = 40.4418, -80.0004  # Pittsburgh center

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

# ------------------- WEATHER API ---------------------
def get_forecast(date_selected):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code != 200:
        return None
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

# ------------------- UI ---------------------
st.set_page_config(layout="wide")
st.title("🚴‍♂️ Trail Usage Predictor (Friends of the Riverfront)")

# ------------------- SINGLE PREDICTION ---------------------
st.subheader("📅 Predict Usage for a Specific Day")

trail = st.selectbox("Choose a trail:", list(proximity_miles.keys()))
date_input = st.date_input("Choose a future date", min_value=datetime.today())

weather = get_forecast(date_input)

if weather:
    st.write("📡 Weather Forecast:", weather)
else:
    st.warning("⚠️ Weather not available — using default values.")
    weather = {
        'Max Temperature (°C)': 20,
        'Precipitation': 0.1,
        'Snowfall': 0.0,
        'Wind speed': 5.0
    }

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
    'lag_1': 0,
    'rolling_mean_3': 0,
    f'Trail_{trail}': 1
}

# Format input
model_input = pd.DataFrame([features])
for col in model.get_booster().feature_names:
    if col not in model_input.columns:
        model_input[col] = 0
model_input = model_input[model.get_booster().feature_names]

if st.button("📈 Predict Usage"):
    pred = model.predict(model_input)[0]
    st.success(f"Predicted usage on {date_input.strftime('%Y-%m-%d')} for {trail}: **{pred:.0f} users**")

# ------------------- 1-YEAR FORECAST ---------------------
st.subheader("📊 1-Year Forecast (Trend)")

if st.button("Show 1-Year Forecast for All Trails"):
    future_days = [datetime.today() + timedelta(days=i) for i in range(365)]
    forecast_rows = []

    for trail in proximity_miles:
        for date in future_days:
            features = {
                'Max Temperature (°C)': 18,
                'Precipitation': 0.1,
                'Snowfall': 0.0,
                'Wind speed': 5.0,
                'Proximity_miles': proximity_miles[trail],
                'weekday': date.weekday(),
                'is_weekend': int(date.weekday() in [5,6]),
                'dayofyear': date.timetuple().tm_yday,
                'month': date.month,
                'lag_1': 0,
                'rolling_mean_3': 0,
                f'Trail_{trail}': 1
            }

            row = pd.DataFrame([features])
            for col in model.get_booster().feature_names:
                if col not in row.columns:
                    row[col] = 0
            row = row[model.get_booster().feature_names]
            prediction = model.predict(row)[0]
            forecast_rows.append((date, trail, prediction))

    forecast_df = pd.DataFrame(forecast_rows, columns=['Date', 'Trail', 'Predicted Usage'])

    for trail in proximity_miles:
        st.write(f"### 📍 {trail}")
        fig, ax = plt.subplots()
        trail_data = forecast_df[forecast_df['Trail'] == trail]
        ax.plot(trail_data['Date'], trail_data['Predicted Usage'], label='Predicted Usage')
        ax.set_ylabel("Users")
        ax.set_xlabel("Date")
        ax.set_title(f"1-Year Trend for {trail}")
        st.pyplot(fig)
