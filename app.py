import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎵 Music Genre Prediction")

st.write("Enter the music features below:")

# Input fields (match training features)
popularity = st.number_input("Popularity", 0, 100, 50)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
duration_ms = st.number_input("Duration (ms)", 10000, 500000, 200000)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
loudness = st.number_input("Loudness", -60.0, 0.0, -10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
tempo = st.number_input("Tempo", 50.0, 250.0, 120.0)
valence = st.slider("Valence", 0.0, 1.0, 0.5)

# Predict button
if st.button("Predict Genre"):

    input_data = np.array([[popularity, acousticness, danceability,
                            duration_ms, energy, instrumentalness,
                            liveness, loudness, speechiness,
                            tempo, valence]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"🎧 Predicted Genre: {prediction[0]}")
