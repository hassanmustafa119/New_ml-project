import gradio as gr
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def predict(gender, race, education, lunch, test_course, reading, writing):

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

    return f"Prediction: {result[0]}"

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["male", "female"], label="Gender"),
        gr.Dropdown(["group A","group B","group C","group D","group E"], label="Race"),
        gr.Dropdown(["high school","associate's","bachelor's","master's"], label="Education"),
        gr.Dropdown(["standard","free/reduced"], label="Lunch"),
        gr.Dropdown(["none","completed"], label="Test Course"),
        gr.Number(label="Reading Score"),
        gr.Number(label="Writing Score"),
    ],
    outputs="text",
    title="Student Performance Predictor"
)

iface.launch()