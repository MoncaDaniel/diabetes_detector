import os
import logging
import joblib
import gradio as gr
import numpy as np
import traceback

# Disable GPU usage (if applicable)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Diabetes Prediction App")
logger.info(f"Python version: {os.sys.version}")
logger.info(f"Installed packages: {os.popen('pip freeze').read()}")

try:
    # Load the model
    model = joblib.load("stacking_model.joblib")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Error loading model", exc_info=True)
    raise e

# Prediction function
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, 
                     Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, GeneticPredisposition):
    try:
        # Create an input array for prediction
        input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, 
                                PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, 
                                DiffWalk, GeneticPredisposition]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Diabetic 🩺" if prediction == 1 else "Not Diabetic 🌞"
        return result
    except Exception as e:
        logger.error("Prediction error:", exc_info=True)
        return "An error occurred during prediction. Please contact support at danielmoncada10@gmail.com."

# Gradio app with a friendly introduction and usage instructions
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🧬 Diabetes Prediction App</h1>")
    gr.Markdown("""
    <div style="text-align: center; font-size: 1.2em; color: #333;">
    <p>This AI tool predicts diabetes risk based on health indicators. Adjust the sliders to reflect your health status for the most accurate prediction! 🌟</p>
    </div>
    
    ### Instructions for Best Results:
    1. 📈 **Be precise** when adjusting the sliders to reflect your current health status.
    2. 🌱 **Lifestyle choices** such as physical activity and diet greatly impact the prediction—set those sliders accurately!
    3. 🧬 **Consider family history** by setting the genetic predisposition slider accordingly.

    ⚠️ **Note**: This tool is for informational purposes only. For medical advice, please consult a healthcare provider.  
    Take charge of your health! 💪🍀  
    -- Daniel
    
    ### Input Your Health Information
    """)
    
    # Input sliders and result output section
    with gr.Row():
        with gr.Column():
            inputs = [
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
            ]
            submit_btn = gr.Button("🔍 Submit for Prediction")
        
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", placeholder="The result will appear here")

    # Trigger prediction on button click
    submit_btn.click(predict_diabetes, inputs=inputs, outputs=result_output)

    # Footer with contact info
    gr.Markdown("""
    ---
    <p style="text-align: center; font-size: 16px;">
        Made with ❤️, data, and code by <span style="color: #228B22; font-weight: bold;">Daniel Moncada León</span>.<br>
        <a href="mailto:danielmoncada10@gmail.com">danielmoncada10@gmail.com</a>
    </p>
    """)

# Launch the Gradio app
demo.launch()
