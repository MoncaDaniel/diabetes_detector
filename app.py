import joblib
import gradio as gr
import numpy as np

# Load the model
model = joblib.load("stacking_model.joblib")

# Define the prediction function
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, 
                     Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, GeneticPredisposition):
    # Create an input array for prediction
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, 
                            PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, 
                            GeneticPredisposition]])
    prediction = model.predict(input_data)[0]
    return "Diabetic 🩺" if prediction == 1 else "Not Diabetic 🌞"

# Define the UI with descriptions and emojis
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(0, 1, step=1, label="High Blood Pressure 🩺"),
        gr.Slider(0, 1, step=1, label="High Cholesterol 🧬"),
        gr.Slider(0, 1, step=1, label="Cholesterol Check ✅"),
        gr.Slider(12, 94, step=1, label="BMI 📏"),
        gr.Slider(0, 1, step=1, label="Smoker 🚬"),
        gr.Slider(0, 1, step=1, label="Stroke 🧠"),
        gr.Slider(0, 1, step=1, label="Heart Disease or Attack ❤️"),
        gr.Slider(0, 1, step=1, label="Physical Activity 🏃"),
        gr.Slider(0, 1, step=1, label="Fruits Intake 🍎"),
        gr.Slider(0, 1, step=1, label="Vegetable Intake 🥦"),
        gr.Slider(0, 1, step=1, label="Heavy Alcohol Consumption 🍻"),
        gr.Slider(1, 5, step=1, label="General Health 🌡️"),
        gr.Slider(0, 30, step=1, label="Mental Health Days 🧘"),
        gr.Slider(0, 30, step=1, label="Physical Health Days 🏋️"),
        gr.Slider(0, 1, step=1, label="Difficulty Walking 🚶"),
        gr.Slider(0, 1, step=1, label="Genetic Predisposition 🧬")
    ],
    outputs="text",
    title="Diabetes Prediction 🧬",
    description="Predict diabetes risk based on health indicators. Adjust sliders to reflect individual factors."
)

# Run the application
if __name__ == "__main__":
    interface.launch()
