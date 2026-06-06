import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")

# Inputs
gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Education", ["high school", "associate's", "bachelor's", "master's"])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_course = st.selectbox("Test Preparation", ["none", "completed"])

reading = st.number_input("Reading Score", min_value=0, max_value=100)
writing = st.number_input("Writing Score", min_value=0, max_value=100)

# Prediction
if st.button("Predict"):
    data = CustomData(
        gender=gender,
        race_ethnicity=race,
        parental_level_of_education=education,
        lunch=lunch,
        test_preparation_course=test_course,
        reading_score=reading,
        writing_score=writing
    )

    df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    st.success(f"Prediction: {result[0]}")