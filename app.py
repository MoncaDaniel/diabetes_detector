import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load("stacking_model.joblib")  # Ensure the path is correct


# Define feature columns based on your dataset
FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", 
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", 
    "Education", "Income", "GeneticPredisposition"
]

# Define prediction function
def predict_diabetes(
    HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
    PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
    NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
    Education, Income, GeneticPredisposition
):
    # Organize inputs in the expected format
    input_data = pd.DataFrame([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                                PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare,
                                NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
                                Education, Income, GeneticPredisposition]], columns=FEATURE_COLUMNS)
    # Make prediction
    prediction = model.predict(input_data)[0]
    return "Diabetes" if prediction == 1 else "No Diabetes"

# Define Gradio interface
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.inputs.Slider(0, 1, step=1, label="High Blood Pressure"),
        gr.inputs.Slider(0, 1, step=1, label="High Cholesterol"),
        gr.inputs.Slider(0, 1, step=1, label="Cholesterol Check"),
        gr.inputs.Slider(10, 50, step=1, label="BMI"),
        gr.inputs.Slider(0, 1, step=1, label="Smoker"),
        gr.inputs.Slider(0, 1, step=1, label="Stroke"),
        gr.inputs.Slider(0, 1, step=1, label="Heart Disease or Attack"),
        gr.inputs.Slider(0, 1, step=1, label="Physical Activity"),
        gr.inputs.Slider(0, 1, step=1, label="Fruits Intake"),
        gr.inputs.Slider(0, 1, step=1, label="Vegetables Intake"),
        gr.inputs.Slider(0, 1, step=1, label="Heavy Alcohol Consumption"),
        gr.inputs.Slider(0, 1, step=1, label="Any Healthcare"),
        gr.inputs.Slider(0, 1, step=1, label="No Doctor Due to Cost"),
        gr.inputs.Slider(1, 5, step=1, label="General Health (1=Excellent, 5=Poor)"),
        gr.inputs.Slider(0, 30, step=1, label="Mental Health (Days of Poor Mental Health)"),
        gr.inputs.Slider(0, 30, step=1, label="Physical Health (Days of Poor Physical Health)"),
        gr.inputs.Slider(0, 1, step=1, label="Difficulty Walking"),
        gr.inputs.Slider(0, 1, step=1, label="Sex (0=Female, 1=Male)"),
        gr.inputs.Slider(18, 120, step=1, label="Age"),
        gr.inputs.Slider(1, 6, step=1, label="Education Level"),
        gr.inputs.Slider(1, 8, step=1, label="Income Level"),
        gr.inputs.Slider(0, 1, step=1, label="Genetic Predisposition"),
    ],
    outputs="text",
    title="Diabetes Prediction",
    description="Predicts diabetes risk based on health indicators"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
