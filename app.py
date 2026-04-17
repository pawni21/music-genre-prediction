import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🎵 Music Genre Prediction")

st.write("Enter song features below to predict the genre")

# Input fields with validation
popularity = st.number_input("Popularity", min_value=0.0, max_value=100.0)
acousticness = st.number_input("Acousticness", min_value=0.0, max_value=1.0)
danceability = st.number_input("Danceability", min_value=0.0, max_value=1.0)
duration_ms = st.number_input("Duration (ms)", min_value=0.0)
energy = st.number_input("Energy", min_value=0.0, max_value=1.0)
instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=1.0)
liveness = st.number_input("Liveness", min_value=0.0, max_value=1.0)
loudness = st.number_input("Loudness", min_value=-60.0, max_value=0.0)
speechiness = st.number_input("Speechiness", min_value=0.0, max_value=1.0)
tempo = st.number_input("Tempo", min_value=0.0, max_value=300.0)
valence = st.number_input("Valence", min_value=0.0, max_value=1.0)

# Predict button
if st.button("Predict Genre"):
    features = np.array([[popularity, acousticness, danceability,
                          duration_ms, energy, instrumentalness,
                          liveness, loudness, speechiness,
                          tempo, valence]])

    features = scaler.transform(features)
    prediction = model.predict(features)

    st.success(f"🎧 Predicted Genre: {prediction[0]}")