import os
import logging
import joblib
import gradio as gr
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"Installed packages: {os.popen('pip freeze').read()}")

# Load the model
try:
    model = joblib.load("stacking_model.joblib")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Error loading model", exc_info=True)
    raise e

# Prediction function with risk levels
def predict_diabetes(GenHlth, HighBP, BMI, HighChol, Age, DiffWalk, PhysHlth, HeartDiseaseorAttack,
                     Stroke, CholCheck, MentHlth, Smoker, GeneticPredisposition):
    # Prepare input data for the model (13 features as expected)
    input_data = np.array([[GenHlth, HighBP, BMI, HighChol, Age, DiffWalk, PhysHlth, HeartDiseaseorAttack,
                            Stroke, CholCheck, MentHlth, Smoker, GeneticPredisposition]])
    
    # Make prediction
    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)[0][1]  # Probability for class 1 (diabetes)

    # Determine risk level based on confidence
    if confidence >= 0.7:
        result = "High Risk of Diabetes"
    elif confidence >= 0.4:
        result = "Risk of Diabetes"
    else:
        result = "Low Risk of Diabetes"
    
    return f"Result: {result} - Confidence: {confidence:.2f}"

# Gradio app with instructions and emojis
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Prediction Tool</h1>")
    gr.Markdown("""
    <div style="text-align: center; font-size: 1.2em; color: #333;">
    <p>This AI tool estimates your risk of diabetes based on health indicators. It provides a risk assessment but does not replace professional medical advice. 💡</p>
    </div>

    ### Instructions for Best Results:
    1. 📋 **Fill in each field** accurately to reflect your health indicators.
    2. 📏 **Check units and ranges** for measurements, such as BMI and age.
    3. 🚨 **Contact a healthcare provider** if you have concerns about your health.

    ⚠️ **Important**: This tool provides risk predictions, not diagnoses. For medical advice, consult a healthcare professional.

    ---
    """)

    gr.Markdown("### Input Your Health Indicators")

    # Health indicator inputs
    with gr.Row():
        with gr.Column():
            GenHlth = gr.Slider(1, 5, step=1, label="General Health 🏥 (1: Excellent - 5: Poor)")
            HighBP = gr.Slider(0, 1, step=1, label="High Blood Pressure 🌡️ (0: No, 1: Yes)")
            BMI = gr.Slider(12, 98, step=1, label="BMI (Body Mass Index) 📊")
            HighChol = gr.Slider(0, 1, step=1, label="High Cholesterol 🧪 (0: No, 1: Yes)")
            Age = gr.Slider(18, 100, step=1, label="Age 🎂")
            DiffWalk = gr.Slider(0, 1, step=1, label="Difficulty Walking 🚶‍♂️ (0: No, 1: Yes)")
            PhysHlth = gr.Slider(0, 30, step=1, label="Physical Health 🤕 (Number of days unwell)")
        
        with gr.Column():
            HeartDiseaseorAttack = gr.Slider(0, 1, step=1, label="Heart Disease or Attack ❤️ (0: No, 1: Yes)")
            Stroke = gr.Slider(0, 1, step=1, label="Stroke 🧠 (0: No, 1: Yes)")
            CholCheck = gr.Slider(0, 1, step=1, label="Cholesterol Check ✅ (0: No, 1: Yes)")
            MentHlth = gr.Slider(0, 30, step=1, label="Mental Health 🧘‍♂️ (Number of days unwell)")
            Smoker = gr.Slider(0, 1, step=1, label="Smoker 🚬 (0: No, 1: Yes)")
            GeneticPredisposition = gr.Slider(0, 1, step=1, label="Genetic Predisposition 🧬 (0: No, 1: Yes)")

    # Prediction button and output
    with gr.Row():
        submit_btn = gr.Button("🔍 Submit for Risk Prediction")
        result_output = gr.Textbox(label="Risk Prediction Result", placeholder="The result will appear here")

    # Link button click to prediction function
    submit_btn.click(
        predict_diabetes,
        inputs=[GenHlth, HighBP, BMI, HighChol, Age, DiffWalk, PhysHlth, HeartDiseaseorAttack,
                Stroke, CholCheck, MentHlth, Smoker, GeneticPredisposition],
        outputs=result_output
    )

    # Footer with contact information
    gr.Markdown("""
    ---
    <p style="text-align: center; font-size: 16px;">
        Made with ❤️, data, and code by <span style="color: #228B22; font-weight: bold;">Daniel Moncada León</span>.<br>
        <a href="mailto:danielmoncada10@gmail.com">danielmoncada10@gmail.com</a>
    </p>
    """)

# Launch the Gradio app
demo.launch(share=True)
