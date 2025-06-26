import streamlit as st
import numpy as np
import pandas as pd
from utils import load_data
from preprocessing import preprocess_data
from sklearn.naive_bayes import GaussianNB

st.title("🔍 Prediksi Manual Status Pasien Cirrhosis")

# Form input user
with st.form("form_pasien"):
    age = st.number_input("Usia", 0, 100, 50)
    bilirubin = st.number_input("Bilirubin", 0.0, 20.0)
    albumin = st.number_input("Albumin", 0.0, 10.0)
    cholesterol = st.number_input("Cholesterol", 0.0, 500.0)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    ascites = st.selectbox("Ascites", ["No", "Yes"])
    hepatomegaly = st.selectbox("Hepatomegaly", ["No", "Yes"])
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Load data dan latih model
    df = load_data("data/cirrhosis.csv")
    X_train, X_test, y_train, y_test, le_status = preprocess_data(df)

    # Ambil fitur kolom dari X_train agar cocok
    colnames = X_train.columns

    # Buat dummy data kosong
    input_row = pd.DataFrame(np.zeros((1, len(colnames))), columns=colnames)

    # Set nilai berdasarkan input (pastikan index-nya cocok!)
    if "Age" in input_row.columns:
        input_row["Age"] = age
    if "Bilirubin" in input_row.columns:
        input_row["Bilirubin"] = bilirubin
    if "Albumin" in input_row.columns:
        input_row["Albumin"] = albumin
    if "Cholesterol" in input_row.columns:
        input_row["Cholesterol"] = cholesterol

    # Cek dan isi fitur biner yang sudah di-label-encode
    if "Gender" in df.columns:
        input_row["Gender"] = 1 if gender == "Male" else 0
    if "Ascites" in df.columns:
        input_row["Ascites"] = 1 if ascites == "Yes" else 0
    if "Hepatomegaly" in df.columns:
        input_row["Hepatomegaly"] = 1 if hepatomegaly == "Yes" else 0

    # Prediksi
    model = GaussianNB()
    model.fit(X_train, y_train)
    pred = model.predict(input_row)
    status = le_status.inverse_transform(pred)[0]

    st.success(f"Hasil prediksi: **{status}**")
