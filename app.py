import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Texas Salary Prediction", layout="centered")
st.title("💼 Texas Employee Salary Prediction")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("salary_data.csv", low_memory=False)
    return df

df = load_data()

# ---------------- CLEAN DATA ----------------
df = df.dropna(subset=["ANNUAL"])
df = df[df["ANNUAL"] > 0]

# Select useful columns
features = [
    "AGENCY NAME",
    "CLASS TITLE",
    "ETHNICITY",
    "GENDER",
    "STATUS",
    "HRS PER WK"
]

target = "ANNUAL"

X = df[features]
y = df[target]

# ---------------- PREPROCESSING ----------------
categorical_features = [
    "AGENCY NAME",
    "CLASS TITLE",
    "ETHNICITY",
    "GENDER",
    "STATUS"
]

numeric_features = ["HRS PER WK"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

# ---------------- MODEL PIPELINE ----------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# ---------------- TRAIN MODEL (CACHED) ----------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    return model

model = train_model()

# ---------------- USER INPUTS ----------------
agency = st.selectbox("Agency Name", df["AGENCY NAME"].unique())
job_title = st.selectbox("Job Title", df["CLASS TITLE"].unique())
ethnicity = st.selectbox("Ethnicity", df["ETHNICITY"].unique())
gender = st.selectbox("Gender", df["GENDER"].unique())
status = st.selectbox("Employment Status", df["STATUS"].unique())
hours = st.number_input("Hours per Week", min_value=1, max_value=60, value=40)

# ---------------- PREDICTION ----------------
if st.button("Predict Annual Salary"):
    input_df = pd.DataFrame([{
        "AGENCY NAME": agency,
        "CLASS TITLE": job_title,
        "ETHNICITY": ethnicity,
        "GENDER": gender,
        "STATUS": status,
        "HRS PER WK": hours
    }])

    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Annual Salary: ₹{prediction[0]:,.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Texas Employee Salary Prediction | Machine Learning Deployment")
