import gradio as gr
import joblib
import pandas as pd

# Load the model
model = joblib.load("stacking_model.joblib")

# Prediction function
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income):
    input_data = pd.DataFrame([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]],
                              columns=['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'])
    prediction = model.predict(input_data)[0]
    return "Diabetes" if prediction == 1 else "No Diabetes"

# Define Gradio interface
inputs = [
    gr.Slider(0, 1, step=1, label="High Blood Pressure"),
    gr.Slider(0, 1, step=1, label="High Cholesterol"),
    gr.Slider(0, 1, step=1, label="Cholesterol Check"),
    gr.Slider(12, 98, step=1, label="BMI"),
    gr.Slider(0, 1, step=1, label="Smoker"),
    gr.Slider(0, 1, step=1, label="Stroke"),
    gr.Slider(0, 1, step=1, label="Heart Disease or Attack"),
    gr.Slider(0, 1, step=1, label="Physical Activity"),
    gr.Slider(0, 1, step=1, label="Fruits"),
    gr.Slider(0, 1, step=1, label="Vegetables"),
    gr.Slider(0, 1, step=1, label="Heavy Alcohol Consumption"),
    gr.Slider(0, 1, step=1, label="Any Healthcare"),
    gr.Slider(0, 1, step=1, label="No Doctor Due to Cost"),
    gr.Slider(1, 5, step=1, label="General Health"),
    gr.Slider(0, 30, step=1, label="Mental Health"),
    gr.Slider(0, 30, step=1, label="Physical Health"),
    gr.Slider(0, 1, step=1, label="Difficulty Walking"),
    gr.Slider(0, 1, step=1, label="Sex (0 for female, 1 for male)"),
    gr.Slider(18, 120, step=1, label="Age"),
    gr.Slider(1, 6, step=1, label="Education"),
    gr.Slider(1, 8, step=1, label="Income"),
]

output = gr.Textbox(label="Prediction")

# Create the Gradio interface
app = gr.Interface(fn=predict_diabetes, inputs=inputs, outputs=output, title="Diabetes Prediction")

# Launch the app
app.launch()
