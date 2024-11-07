# Diabetes Prediction Model

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Instructions for Use](#instructions-for-use)
- [Documentation](#documentation)

## Introduction
This project is a machine learning model designed to predict the likelihood of diabetes based on health indicators. Leveraging a clean, balanced dataset, the model provides a robust risk assessment for diabetes and prediabetes. This application aims to assist healthcare professionals and individuals in identifying early risk factors, allowing for preventive action.

## Dataset Description
The dataset used for this project is the **BRFSS2015 Health Indicators** dataset, sourced from the Centers for Disease Control and Prevention (CDC). It contains 70,692 survey responses with an even split of non-diabetes and diabetes/prediabetes cases. The dataset includes 21 health-related features, such as:

- **General Health (GenHlth)**
- **High Blood Pressure (HighBP)**
- **BMI**
- **High Cholesterol (HighChol)**
- **Age**
- **Difficulty Walking (DiffWalk)**
- **Physical Health Issues (PhysHlth)**
- **Heart Disease/Stroke History**
- **Cholesterol Check (CholCheck)**
- **Mental Health Issues (MentHlth)**
- **Smoking Status**

Since the dataset lacked genetic predisposition data, synthetic data was generated to approximate this feature based on realistic distributions.

## Model Performance
The model uses a stacking ensemble approach that combines several classifiers to improve prediction accuracy. Key components of the ensemble include logistic regression, decision tree, random forest, and gradient boosting. **Summary of performance metrics**:

- **Cross-Validation Score**: 0.8449
- **Test Accuracy**: 0.8394

### Classification Report

| Metric       | Class 0 (No Diabetes) | Class 1 (Diabetes) |
|--------------|------------------------|---------------------|
| Precision    | 0.82                  | 0.86               |
| Recall       | 0.86                  | 0.82               |
| F1 Score     | 0.84                  | 0.84               |

These metrics indicate the model’s ability to correctly identify diabetes cases while minimizing false positives.

## Quick Start
To set up the project and run the model, follow these steps:

### Prerequisites
- **Python 3.7+**
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `gradio`, `joblib`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MoncaDaniel/diabetes-prediction.git
   cd diabetes-prediction
