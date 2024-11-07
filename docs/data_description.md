# Data Description

## Dataset Overview
The dataset used for this project is the **BRFSS2015 Health Indicators** dataset, obtained from the Centers for Disease Control and Prevention (CDC). This dataset includes health-related information from 70,692 survey respondents, balanced evenly between non-diabetes and diabetes/prediabetes cases. It’s designed to support predictive analysis for diabetes risk.

### Key Features
The dataset contains 21 features related to health, lifestyle, and medical history. The following features were selected for model training based on their relevance to diabetes risk:

- **GenHlth**: Self-reported general health status (scale 1-5)
- **HighBP**: Indicator of high blood pressure (binary)
- **BMI**: Body Mass Index, calculated from weight and height
- **HighChol**: Indicator of high cholesterol levels (binary)
- **Age**: Age group of the respondent
- **DiffWalk**: Difficulty walking or climbing stairs (binary)
- **PhysHlth**: Number of days with physical health issues in the past 30 days
- **HeartDiseaseorAttack**: History of heart disease or heart attack (binary)
- **Stroke**: History of stroke (binary)
- **CholCheck**: Has had a cholesterol check in the past five years (binary)
- **MentHlth**: Number of days with mental health issues in the past 30 days
- **Smoker**: Smoking status (binary)

The target variable, `Diabetes_binary`, categorizes respondents into two classes:
- **0**: No diabetes
- **1**: Prediabetes or diabetes

### Synthetic Data Generation
The original dataset does not include genetic predisposition information, a relevant factor for diabetes prediction. To address this, a synthetic feature called **GeneticPredisposition** was created to estimate genetic risk based on real-world prevalence rates. The following steps outline the synthetic data generation process:

1. **Real-World Data Reference**: Established general population prevalence rates for genetic predisposition to diabetes.
2. **Random Sampling**: For each respondent, a synthetic genetic predisposition score was generated using a probability distribution informed by real-world data.
3. **Integration**: This synthetic feature was added to the dataset as a new column, used during model training to improve prediction accuracy.

### Data Preprocessing
To prepare the data for modeling, several preprocessing steps were implemented:

1. **Standardization and Normalization**:
   - **BMI** and **Age** values were standardized to ensure a consistent scale across features.
   - Normalization was applied to features with skewed distributions to prevent model bias.

2. **Outlier Detection and Handling**:
   - Outliers were identified using interquartile ranges (IQR) for continuous variables like **BMI** and **PhysHlth**.
   - Outliers were adjusted to be within acceptable ranges to avoid skewing the model’s learning process.

3. **Handling Missing Values**:
   - The dataset was already clean, with no missing values for the selected features. However, minor data inconsistencies were addressed by ensuring all binary fields were encoded as 0 or 1.

4. **Feature Engineering**:
   - **GeneticPredisposition**: Integrated the synthetic genetic feature as an additional input.
   - **Binary Encoding**: Categorical variables were binarized where applicable (e.g., Smoker status).
   - **Aggregated Health Score**: Created an aggregated health score by combining features related to physical and mental health, providing an overall health indicator for each respondent.

### Summary
The dataset was prepared with several essential transformations to ensure optimal model training. By focusing on features relevant to diabetes risk and incorporating synthetic genetic data, the dataset provides a solid foundation for building a reliable predictive model.

For further details on data handling and processing, refer to the code comments in `data_preprocessing.py`.

