import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load the model
stacking_model = joblib.load("stacking_model.joblib")

# Define the prediction function
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                     PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
                     NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income):
    
    # Arrange input in the correct order
    features = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                          PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
                          NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])
    
    # Predict using the stacking model
    prediction = stacking_model.predict(features)
    
    # Interpret the prediction
    return "Diabetic" if prediction[0] == 1 else "Non-diabetic"

# Create the Gradio Interface with a radio-style layout
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.inputs.Radio(choices=[0, 1], label="HighBP"),
        gr.inputs.Radio(choices=[0, 1], label="HighChol"),
        gr.inputs.Radio(choices=[0, 1], label="CholCheck"),
        gr.inputs.Slider(10, 100, step=1, label="BMI"),
        gr.inputs.Radio(choices=[0, 1], label="Smoker"),
        gr.inputs.Radio(choices=[0, 1], label="Stroke"),
        gr.inputs.Radio(choices=[0, 1], label="HeartDiseaseorAttack"),
        gr.inputs.Radio(choices=[0, 1], label="PhysActivity"),
        gr.inputs.Radio(choices=[0, 1], label="Fruits"),
        gr.inputs.Radio(choices=[0, 1], label="Veggies"),
        gr.inputs.Radio(choices=[0, 1], label="HvyAlcoholConsump"),
        gr.inputs.Radio(choices=[0, 1], label="AnyHealthcare"),
        gr.inputs.Radio(choices=[0, 1], label="NoDocbcCost"),
        gr.inputs.Slider(1, 5, step=1, label="GenHlth"),
        gr.inputs.Slider(0, 30, step=1, label="MentHlth"),
        gr.inputs.Slider(0, 30, step=1, label="PhysHlth"),
        gr.inputs.Radio(choices=[0, 1], label="DiffWalk"),
        gr.inputs.Radio(choices=[0, 1], label="Sex"),
        gr.inputs.Slider(18, 100, step=1, label="Age"),
        gr.inputs.Slider(1, 6, step=1, label="Education"),
        gr.inputs.Slider(1, 8, step=1, label="Income")
    ],
    outputs="text",
    title="Diabetes Prediction Model",
    description="Predicts diabetes based on various health indicators.",
)

# Launch the app
iface.launch()
