import pandas as pd
import streamlit as st

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# 4. preprocessing.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(df):
    df = df.copy()
    df.drop(columns=["ID"], errors="ignore", inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop("Status", axis=1)
    y = df["Status"]
    le_status = LabelEncoder()
    y = le_status.fit_transform(y)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, le_status
