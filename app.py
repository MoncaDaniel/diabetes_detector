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

# Prediction function with risk levels
def predict_diabetes(HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                     PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk, GeneticPredisposition):
    # Prepare input data for the model (13 features as expected)
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                            PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk, GeneticPredisposition]])
    
    # Predict diabetes risk
    prediction = model.predict(input_data)
    risk_level = "Low Risk" if prediction[0] == 0 else "Risk" if prediction[0] == 1 else "High Risk"
    return f"Result: {risk_level} of Diabetes"

# Gradio app setup with instructions and a friendly interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Prediction App</h1>")
    gr.Markdown("""
    <div style="text-align: center; font-size: 1.2em; color: #333;">
    <p>Welcome to the Diabetes Risk Prediction App! This tool estimates your risk of diabetes based on key health indicators.</p>
    </div>
    
    ### Instructions for Best Results:
    1. 📋 **Enter accurate values** for each category to get the most reliable prediction.
    2. ⚖️ **Maintain a healthy weight** and lifestyle for the best health outcomes.
    3. 🍏 **Healthy lifestyle choices** are recommended regardless of the prediction.

    ⚠️ **Note**: This tool is for informational purposes only and should not replace professional medical advice.
    If you have any health concerns, consult a healthcare provider.

    -- Daniel
    
    """)

    # Feature inputs with explanations and emojis
    with gr.Row():
        HighBP = gr.Slider(0, 1, step=1, label="High Blood Pressure 🩸 - Indicate 1 if diagnosed with high blood pressure")
        HighChol = gr.Slider(0, 1, step=1, label="High Cholesterol 🧬 - Indicate 1 if diagnosed with high cholesterol")
        CholCheck = gr.Slider(0, 1, step=1, label="Cholesterol Check 🩺 - 1 if you had a cholesterol check in the past year")
        BMI = gr.Slider(12, 94, step=1, label="BMI 📏 - Enter your Body Mass Index")
        Smoker = gr.Slider(0, 1, step=1, label="Smoker 🚬 - 1 if you currently smoke")
        Stroke = gr.Slider(0, 1, step=1, label="Stroke 🧠 - 1 if you have had a stroke")
        HeartDiseaseorAttack = gr.Slider(0, 1, step=1, label="Heart Disease ❤️ - 1 if diagnosed with heart disease or heart attack")
        PhysActivity = gr.Slider(0, 1, step=1, label="Physical Activity 🏃‍♂️ - 1 if you engage in physical activity")
        Fruits = gr.Slider(0, 1, step=1, label="Fruits Intake 🍎 - 1 if you eat fruits at least once per day")
        Veggies = gr.Slider(0, 1, step=1, label="Vegetable Intake 🥦 - 1 if you eat vegetables at least once per day")
        HvyAlcoholConsump = gr.Slider(0, 1, step=1, label="Heavy Alcohol Consumption 🍻 - 1 if you have heavy alcohol consumption")
        GenHlth = gr.Slider(1, 5, step=1, label="General Health 🏥 - Rate your general health from 1 (excellent) to 5 (poor)")
        DiffWalk = gr.Slider(0, 1, step=1, label="Difficulty Walking 🚶‍♀️ - 1 if you have difficulty walking")
        GeneticPredisposition = gr.Slider(0, 1, step=1, label="Genetic Predisposition 🧬 - 1 if you have family history of diabetes")

    gr.Markdown("### Predict Your Diabetes Risk")
    
    # Prediction output section
    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("🔍 Submit for Analysis")
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", placeholder="The result will appear here")

    # Trigger prediction on button click
    submit_btn.click(predict_diabetes, inputs=[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                                               PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, DiffWalk, GeneticPredisposition],
                     outputs=result_output)

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
