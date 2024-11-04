import gradio as gr
import joblib
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the model
    model = joblib.load("stacking_model.joblib")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Error loading model", exc_info=True)
    raise e

# Only select the features the model expects
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                     PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk):
    # Create input array with only the required features
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                            PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk]])
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return f"Result: {result}"

# Gradio app with feature explanations
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🩺 Diabetes Prediction App</h1>")
    gr.Markdown("""
    This AI tool predicts diabetes risk based on health indicators. Please enter values for each category below:
    """)

    with gr.Row():
        HighBP = gr.Slider(0, 1, step=1, label="High Blood Pressure 🩸")
        HighChol = gr.Slider(0, 1, step=1, label="High Cholesterol 🧬")
        CholCheck = gr.Slider(0, 1, step=1, label="Cholesterol Check 🩺")
        BMI = gr.Slider(12, 94, step=1, label="BMI 📏")
        Smoker = gr.Slider(0, 1, step=1, label="Smoker 🚬")
        Stroke = gr.Slider(0, 1, step=1, label="Stroke 🧠")
        HeartDiseaseorAttack = gr.Slider(0, 1, step=1, label="Heart Disease ❤️")
        PhysActivity = gr.Slider(0, 1, step=1, label="Physical Activity 🏃‍♂️")
        Fruits = gr.Slider(0, 1, step=1, label="Fruits Intake 🍎")
        Veggies = gr.Slider(0, 1, step=1, label="Vegetable Intake 🥦")
        HvyAlcoholConsump = gr.Slider(0, 1, step=1, label="Heavy Alcohol Consumption 🍻")
        GenHlth = gr.Slider(1, 5, step=1, label="General Health 🏥")
        DiffWalk = gr.Slider(0, 1, step=1, label="Difficulty Walking 🚶‍♀️")

    result_output = gr.Textbox(label="Prediction Result")
    submit_btn = gr.Button("🔍 Submit")

    # Trigger prediction on button click
    submit_btn.click(predict_diabetes, inputs=[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                                               PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk],
                     outputs=result_output)

    gr.Markdown("""
    ---
    <p style="text-align: center; font-size: 16px;">
        Made with ❤️, data, and code by Daniel Moncada León.<br>
        <a href="mailto:danielmoncada10@gmail.com">danielmoncada10@gmail.com</a>
    </p>
    """)

# Launch the Gradio app
demo.launch()
