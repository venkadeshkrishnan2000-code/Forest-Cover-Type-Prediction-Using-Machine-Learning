import streamlit as st
import numpy as np
import joblib

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

st.title("🌲 EcoType Forest Cover Predictor")
st.write("Enter the environmental details:")

# Generic label function
def categorize(value, ranges, labels):
    for r, l in zip(ranges, labels):
        if value < r:
            return l
    return labels[-1]

# Elevation
Elevation = st.number_input("⛰️ Elevation (meters)", 0, 5000, 0, 50)
st.write("📍 Elevation Zone:",
         categorize(Elevation, [1000, 2000, 3000],
                    ["Low 🌾", "Moderate 🌿", "High 🌲", "Very High 🏔️"]))

# Water distance
Horizontal_Distance_To_Hydrology = st.number_input("💧 Distance to Water", 0, 10000, 0, 10)
st.write("📍 Water Proximity:",
         categorize(Horizontal_Distance_To_Hydrology, [100, 500, 2000],
                    ["Very Close 💧", "Moderately Close 🌊", "Far 🌵", "Very Far 🏜️"]))

# Vertical distance
Vertical_Distance_To_Hydrology = st.number_input("💧 Vertical Distance to Water", -500, 500, 0)
st.write("📍 Position:",
         "Below 🌊" if Vertical_Distance_To_Hydrology < 0 else
         "At 💧" if Vertical_Distance_To_Hydrology == 0 else "Above ⛰️")

# Road distance
Horizontal_Distance_To_Roadways = st.number_input("🛣️ Distance to Roadways", 0, 10000, 0, 10)
st.write("📍 Road Proximity:",
         categorize(Horizontal_Distance_To_Roadways, [100, 500, 2000],
                    ["Very Close 🚗", "Moderate 🛣️", "Far 🌄", "Remote 🌲"]))

# Fire distance
Horizontal_Distance_To_Fire_Points = st.number_input("🔥 Distance to Fire Points", 0, 10000, 0, 10)
st.write("🔥 Fire Risk:",
         categorize(Horizontal_Distance_To_Fire_Points, [200, 1000, 3000],
                    ["High 🔥", "Moderate ⚠️", "Low 🌲", "Very Low 🌿"]))

# Wilderness
Wilderness_Area = st.selectbox("🌲 Wilderness Area", [1, 2, 3, 4])
st.write("📍 Area Info:",
         ["Dense 🌲", "Alpine ⛰️", "Rocky 🏔️", "River 🌊"][Wilderness_Area-1])

# Soil
soil_map = {f"Soil Type {i}": i for i in range(1, 41)}
selected_soil = st.selectbox("🌱 Soil Type", list(soil_map))
Soil_Type = soil_map[selected_soil]

soil_num = Soil_Type
st.write("📍 Soil Info:",
         categorize(soil_num, [5, 10, 20, 30],
                    ["Rocky 🪨", "Sandy 🌵", "Loamy 🌱", "Clay 🏺", "Organic 🌿"]))

# Aspect
Aspect = st.slider("Aspect (Degrees)", 0, 360, 0)
dirs = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"]
st.write("📍 Direction:", dirs[int((Aspect+22.5)//45) % 8])

# Slope
Slope = st.slider("Slope", 0, 60, 0)
st.write("⛰️ Terrain:",
         categorize(Slope, [5, 15, 30, 45],
                    ["Flat", "Gentle", "Moderate", "Steep", "Very Steep"]))

# Light levels (single reusable logic)
def light(val):
    return categorize(val, [50, 100, 180, 220],
                      ["Very Dark 🌑", "Low 🌥️", "Moderate 🌤️", "Bright ☀️", "Very Bright 🔆"])

Hillshade_9am = st.slider("Hillshade 9AM", 0, 255, 0)
st.write("💡", light(Hillshade_9am))

Hillshade_Noon = st.slider("Hillshade Noon", 0, 255, 0)
st.write("💡", light(Hillshade_Noon))

Hillshade_3pm = st.slider("Hillshade 3PM", 0, 255, 0)
st.write("💡", light(Hillshade_3pm))

# Prediction
if st.button("Predict"):
    features = np.array([[Elevation, Aspect, Slope,
                          Horizontal_Distance_To_Hydrology,
                          Vertical_Distance_To_Hydrology,
                          Horizontal_Distance_To_Roadways,
                          Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
                          Horizontal_Distance_To_Fire_Points,
                          Wilderness_Area, Soil_Type]])

    pred = model.predict(scaler.transform(features))
    result = target_encoder.inverse_transform(pred)
    st.success(f"🌳 Predicted Forest Type: {result[0]}")
    