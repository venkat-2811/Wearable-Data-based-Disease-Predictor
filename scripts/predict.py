import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Load model artifacts
with open('../models/disease_predictor.pkl', 'rb') as file:
    model_artifacts = pickle.load(file)

# Load disease mappings
disease_mappings = pd.read_csv('../data/FINAL-LABELS-DATASET.csv')
disease_mappings.fillna("Unknown", inplace=True)
disease_mappings = disease_mappings[disease_mappings['Disease_Label'] != "Unknown"]
disease_mappings['Encoded_Label'] = model_artifacts['encoders']['disease'].transform(disease_mappings['Disease_Label'])

# Define helper functions
def preprocess_input(input_data, scaler, encoders, feature_names):
    """Preprocesses the input data for prediction."""
    missing_features = [col for col in feature_names if col not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    if 'Activity_Level' in input_data.columns and 'activity' in encoders:
        input_data['Activity_Level'] = encoders['activity'].transform(input_data['Activity_Level'])

    data = input_data[feature_names]
    data_scaled = scaler.transform(data)
    return pd.DataFrame(data_scaled, columns=feature_names)

def predict_disease(input_data, model_artifacts, disease_mappings):
    """Predicts diseases, provides general recommendations, and makes future predictions."""
    # Expanded normal ranges
    normal_ranges = {
        'Heart_Rate': (50, 120),
        'HRV': (10, 100),
        'ECG': (-2, 2),
        'SpO2': (90, 100),
        'Respiration_Rate': (10, 25),
        'Temperature': (35.5, 38.0),
        'Sleep_Duration': (5, 9),
        'REM_Sleep': (0.5, 3),
        'Step_Count': (2000, 15000),
        'EDA': (0, 10),
        'Blood_Pressure_Systolic': (80, 140),
        'Blood_Pressure_Diastolic': (50, 90),
        'Blood_Glucose': (60, 180)
    }

    # General health recommendations
    general_disease_info = {
        'Heart Disease': {
            "Focus On": "Heart Rate, HRV, Blood Pressure",
            "Recommendations": "Manage stress with mindfulness or relaxation techniques. Exercise regularly (e.g., 30 minutes of moderate activity daily). Avoid smoking and maintain a balanced diet."
        },
        'Breathing Disorders': {
            "Focus On": "Respiration Rate, SpO2",
            "Recommendations": "Practice breathing exercises. Avoid exposure to allergens or pollutants. Consult a doctor for chronic symptoms."
        },
        'Diabetes': {
            "Focus On": "Blood Glucose",
            "Recommendations": "Monitor carbohydrate intake and opt for low-glycemic-index foods. Maintain a healthy weight through balanced diet and exercise. Check blood sugar levels regularly."
        },
        'Sleep Disorders': {
            "Focus On": "Sleep Duration, REM Sleep",
            "Recommendations": "Establish a consistent sleep schedule. Avoid caffeine or heavy meals before bedtime. Keep the sleeping environment quiet and dark."
        },
        'Fever or Hypothermia': {
            "Focus On": "Temperature",
            "Recommendations": "If feverish, stay hydrated and rest. If feeling cold, ensure adequate clothing and heating. Seek medical attention for prolonged symptoms."
        },
        'Sedentary Lifestyle': {
            "Focus On": "Step Count, Activity Level",
            "Recommendations": "Aim for at least 10,000 steps daily. Include strength and flexibility exercises in your routine. Avoid prolonged sitting; take regular breaks to stretch."
        },
        'Chronic Stress or Anxiety': {
            "Focus On": "EDA (Electrodermal Activity), HRV",
            "Recommendations": "Practice mindfulness, meditation, or yoga. Prioritize time management and healthy social interactions. Seek therapy or counseling if necessary."
        },
        'Hypertension or Hypotension': {
            "Focus On": "Blood Pressure (Systolic and Diastolic)",
            "Recommendations": "Reduce sodium intake and consume potassium-rich foods. Limit alcohol and avoid tobacco products. Monitor blood pressure regularly."
        },
        'Hypoxemia': {
            "Focus On": "SpO2",
            "Recommendations": "Use a pulse oximeter to monitor SpO2 levels. Ensure proper ventilation in your environment. Seek immediate medical attention if levels fall below 90%."
        },
        'Neuropathy or Seizures': {
            "Focus On": "ECG, HRV",
            "Recommendations": "Avoid triggers such as excessive stress or lack of sleep. Stay hydrated and maintain a healthy diet. Consult a neurologist if symptoms persist."
        }
    }

    # Results list
    results = []
    
    # Evaluate each row in the input data
    for idx, row in input_data.iterrows():
        abnormal_metrics = []
        for feature, (low, high) in normal_ranges.items():
            if feature in row and not (low <= row[feature] <= high):
                abnormal_metrics.append(feature)

        if not abnormal_metrics:  # All values are in normal range
            results.append({
                "Disease": "No Disease - You are Healthy",
                "Focus On": "All values are in range",
                "Recommendations": "You are doing well with your routine",
                "Probability": 1.0
            })
        else:  # Map abnormalities to general disease info
            if 'Blood_Pressure_Systolic' in abnormal_metrics or 'Blood_Pressure_Diastolic' in abnormal_metrics:
                disease_name = "Hypertension or Hypotension"
            elif 'Blood_Glucose' in abnormal_metrics:
                disease_name = "Diabetes"
            elif 'Respiration_Rate' in abnormal_metrics or 'SpO2' in abnormal_metrics:
                disease_name = "Breathing Disorders"
            elif 'Heart_Rate' in abnormal_metrics or 'ECG' in abnormal_metrics:
                disease_name = "Heart Disease"
            elif 'Temperature' in abnormal_metrics:
                disease_name = "Fever or Hypothermia"
            elif 'Step_Count' in abnormal_metrics or 'Activity_Level' in abnormal_metrics:
                disease_name = "Sedentary Lifestyle"
            elif 'EDA' in abnormal_metrics or 'HRV' in abnormal_metrics:
                disease_name = "Chronic Stress or Anxiety"
            elif 'SpO2' in abnormal_metrics:
                disease_name = "Hypoxemia"
            elif 'ECG' in abnormal_metrics or 'HRV' in abnormal_metrics:
                disease_name = "Neuropathy or Seizures"
            else:
                disease_name = "General Health Concern"

            disease_info = general_disease_info.get(disease_name, {"Focus On": "General Health", "Recommendations": "Consult a healthcare provider."})
            results.append({
                "Disease": disease_name,
                "Focus On": disease_info["Focus On"],
                "Recommendations": disease_info["Recommendations"],
                "Probability": 0.85  # Assign a generic probability for non-model predictions
            })

    return results

# Function to plot time series data
def plot_time_series(data, feature, title):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y=feature, data=data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot distribution of a feature
def plot_distribution(data, feature, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot correlation heatmap
def plot_correlation_heatmap(data, title):
    plt.figure(figsize=(12, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit App
st.title("Health-Sphere: Disease Prediction & Health Monitoring")

# Sidebar for user options
st.sidebar.title("Options")
menu = st.sidebar.radio("Menu", ["HEALTH ANALYSIS ON FILE-DATA", "HEALTH ANALYSIS ON GOOGLE-FIT DATA",])

if menu == "HEALTH ANALYSIS ON FILE-DATA":
    st.subheader("Upload Your Health Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        
        try:
            # Ensure the file contains a 'Date' column
            if 'Date' not in input_data.columns:
                raise ValueError("The uploaded file must contain a 'Date' column.")

            # Convert the 'Date' column to datetime
            input_data['Date'] = pd.to_datetime(input_data['Date'])
            
            # Determine the next date for prediction
            latest_date = input_data['Date'].max()
            next_date = latest_date + pd.Timedelta(days=1)

            # Use the last row's data (excluding 'Date') for prediction
            latest_data = input_data.loc[input_data['Date'] == latest_date].iloc[0]
            latest_data = latest_data.drop(labels='Date')

            # Create a new DataFrame for the next day's prediction
            next_day_data = latest_data.to_frame().T
            next_day_data['Date'] = next_date

            # Predict disease for the next day
            predictions = predict_disease(next_day_data, model_artifacts, disease_mappings)

            st.subheader(f"Predictions for {next_date.strftime('%Y-%m-%d')}")
            for idx, pred in enumerate(predictions):
                st.write(f"Prediction {idx + 1}:")
                st.write(f"- **Disease**: {pred['Disease']}")
                st.write(f"- **Focus On**: {pred['Focus On']}")
                st.write(f"- **Recommendations**: {pred['Recommendations']}")
                st.write(f"- **Probability**: {pred['Probability']:.2f}")
                st.write("---")

            st.subheader("Data Visualizations")
            st.write("### Time Series Plots")
            for feature in input_data.columns:
                if feature != 'Date':
                    plot_time_series(input_data, feature, f'Time Series of {feature}')

            st.write("### Feature Distributions")
            for feature in input_data.columns:
                if feature != 'Date':
                    plot_distribution(input_data, feature, f'Distribution of {feature}')

        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "HEALTH ANALYSIS ON GOOGLE-FIT DATA":
    st.subheader("Filter Predictions by Date")
    start_date = st.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.date_input("End Date", datetime(2024, 11, 16))
    
    # Convert to datetime64[ns] for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            synthetic_data = pd.read_csv("../user_data/google-fit-data-venkata-karthik-sai.csv")
            synthetic_data["Date"] = pd.to_datetime(synthetic_data["Date"])  # Ensure 'Date' column is datetime
            filtered_data = synthetic_data[(synthetic_data["Date"] >= start_date) & (synthetic_data["Date"] <= end_date)]
            
            if filtered_data.empty:
                st.warning("No data available for the selected date range.")
            else:
                # Future Prediction
                next_date = end_date + pd.Timedelta(days=1)
                median_row = filtered_data.median(numeric_only=True)
                median_row['Date'] = next_date
                next_day_data = pd.DataFrame([median_row])

                # Predict for the next day
                next_day_predictions = predict_disease(next_day_data, model_artifacts, disease_mappings)
                st.subheader(f"Predictions for {next_date.strftime('%Y-%m-%d')}")
                for idx, pred in enumerate(next_day_predictions):
                    st.write(f"Future Prediction {idx + 1}:")
                    st.write(f"- **Disease**: {pred['Disease']}")
                    st.write(f"- **Focus On**: {pred['Focus On']}")
                    st.write(f"- **Recommendations**: {pred['Recommendations']}")
                    st.write(f"- **Probability**: {pred['Probability']:.2f}")
                    st.write("---")

                st.subheader("Data Visualizations")
                st.write("### Time Series Plots")
                for feature in filtered_data.columns:
                    if feature != 'Date':
                        plot_time_series(filtered_data, feature, f'Time Series of {feature}')

                st.write("### Feature Distributions")
                for feature in filtered_data.columns:
                    if feature != 'Date':
                        plot_distribution(filtered_data, feature, f'Distribution of {feature}')


        except Exception as e:
            st.error(f"Error: {e}")
