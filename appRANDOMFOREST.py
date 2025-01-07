import streamlit as st
import pandas as pd
import joblib


# Load model dan label encoder
model = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder_RFC.pkl')

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi algoritma RF", layout="centered")
st.title("Prediksi Fruit Random Forest")

st.markdown(
    "Memprediksi data baru dengan algoritma RANDOM FOREST"
)

# Input data pengguna
st.header("Masukkan Data Fruit")
diameter = st.number_input("diameter", value=0.0, format="%.2f")
weight = st.number_input("weight", value=0.0, format="%.2f")
red = st.number_input("red", value=0.0, format="%.2f")
green = st.number_input("green", value=0.0, format="%.2f")
blue = st.number_input("blue", value=0.0, format="%.2f")

# Button untuk prediksi
if st.button("Prediksi"):
    # Data baru untuk prediksi
    data_baru = pd.DataFrame({
        "diameter": [diameter],
        "weight": [weight],
        "red": [red],
        "green": [green],
        "blue": [blue]
    })

    # Prediksi dan konversi label ke kategori
    prediksi = model.predict(data_baru)
    kategori = label_encoder.inverse_transform(prediksi)

    # Tampilkan hasil
    st.success(f"Kategori buah yang diprediksi: **{kategori[0]}**")
