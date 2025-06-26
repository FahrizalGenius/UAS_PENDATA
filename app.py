import streamlit as st
from utils import load_data
from preprocessing import preprocess_data
from modeling import train_and_predict
import streamlit.components.v1 as components

st.set_page_config(page_title="Cirrhosis Predictor", layout="wide")

with open("templates/homepage.html", "r") as f:
    html = f.read()
components.html(html, height=180)

st.subheader("Dataset")
df = load_data("data/cirrhosis.csv")
st.dataframe(df.head())

X_train, X_test, y_train, y_test, le_status = preprocess_data(df)

st.subheader("Evaluasi Model")
model_name = st.selectbox("Pilih model", ["SVM", "KNN", "Naive Bayes"])
result = train_and_predict(model_name, X_train, X_test, y_train, y_test, le_status)
st.text("Classification Report:")
st.text(result["report"])
st.pyplot(result["cm_plot"])