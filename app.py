import gradio as gr
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load("stacking_model.joblib")

# Define the prediction function
def predict_diabetes(high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                     fruits, veggies, hvy_alcohol, any_healthcare, no_docbc_cost, gen_health, ment_health,
                     phys_health, diff_walk):
    features = np.array([[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, phys_activity,
                          fruits, veggies, hvy_alcohol, any_healthcare, no_docbc_cost, gen_health, ment_health,
                          phys_health, diff_walk]])
    prediction = model.predict(features)
    return "🟢 Diabetes not detected" if prediction[0] == 0 else "🔴 Diabetes detected"

# Set up the Gradio interface with emojis and explanations
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Diabetes Prediction Application\nWelcome! Please fill out the following health indicators, and we'll predict your risk of diabetes. ")

    with gr.Row():
        high_bp = gr.Radio(choices=[1, 0], label="High Blood Pressure 💓", info="Do you have high blood pressure?")
        high_chol = gr.Radio(choices=[1, 0], label="High Cholesterol 🧬", info="Have you been diagnosed with high cholesterol?")
        chol_check = gr.Radio(choices=[1, 0], label="Cholesterol Check 🧪", info="Have you had a cholesterol check in the past year?")
        bmi = gr.Slider(12, 98, step=1, label="Body Mass Index (BMI) ⚖️", info="What is your Body Mass Index (BMI)?")

    with gr.Row():
        smoker = gr.Radio(choices=[1, 0], label="Smoker 🚬", info="Do you currently smoke?")
        stroke = gr.Radio(choices=[1, 0], label="Stroke 🧠", info="Have you ever had a stroke?")
        heart_disease = gr.Radio(choices=[1, 0], label="Heart Disease or Attack 💔", info="Have you had a heart attack or other heart disease?")
        phys_activity = gr.Radio(choices=[1, 0], label="Physical Activity 🏃", info="Do you engage in regular physical activity?")

    with gr.Row():
        fruits = gr.Radio(choices=[1, 0], label="Fruits 🍎", info="Do you eat fruits daily?")
        veggies = gr.Radio(choices=[1, 0], label="Vegetables 🥦", info="Do you eat vegetables daily?")
        hvy_alcohol = gr.Radio(choices=[1, 0], label="Heavy Alcohol Consumption 🍷", info="Do you consume alcohol heavily?")
        any_healthcare = gr.Radio(choices=[1, 0], label="Any Healthcare 🏥", info="Do you have access to healthcare?")

    with gr.Row():
        no_docbc_cost = gr.Radio(choices=[1, 0], label="No Doctor Due to Cost 💵", info="Did you skip doctor visits due to cost?")
        gen_health = gr.Slider(1, 5, step=1, label="General Health 🌟", info="How would you rate your general health?")
        ment_health = gr.Slider(0, 30, step=1, label="Mental Health 🤯", info="Number of days with mental health issues in the past month")
        phys_health = gr.Slider(0, 30, step=1, label="Physical Health 🏋️", info="Number of days with physical health issues in the past month")
        diff_walk = gr.Radio(choices=[1, 0], label="Difficulty Walking 🚶‍♂️", info="Do you have difficulty walking?")

    submit_btn = gr.Button("Predict Diabetes Risk 🧮")
    output = gr.Textbox(label="Prediction Result")

    submit_btn.click(predict_diabetes, inputs=[high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
                                               phys_activity, fruits, veggies, hvy_alcohol, any_healthcare,
                                               no_docbc_cost, gen_health, ment_health, phys_health, diff_walk],
                     outputs=output)

demo.launch()
