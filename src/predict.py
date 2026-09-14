import os
import joblib
import pandas as pd


# -----------------------------
# File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "final_attrition_model.pkl")


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load(MODEL_PATH)

print("Final attrition model loaded successfully.")


# -----------------------------
# Sample employee data
# -----------------------------
employee_data = {
    "Age": [30],
    "BusinessTravel": ["Travel_Rarely"],
    "DailyRate": [800],
    "Department": ["Research & Development"],
    "DistanceFromHome": [10],
    "Education": [3],
    "EducationField": ["Life Sciences"],
    "EnvironmentSatisfaction": [3],
    "Gender": ["Male"],
    "JobInvolvement": [3],
    "JobLevel": [2],
    "JobRole": ["Research Scientist"],
    "JobSatisfaction": [3],
    "MaritalStatus": ["Single"],
    "MonthlyIncome": [4000],
    "HourlyRate": [65],
    "MonthlyRate": [15000],
    "NumCompaniesWorked": [2],
    "OverTime": ["Yes"],
    "PercentSalaryHike": [15],
    "PerformanceRating": [3],
    "RelationshipSatisfaction": [3],
    "StockOptionLevel": [0],
    "TotalWorkingYears": [8],
    "TrainingTimesLastYear": [3],
    "WorkLifeBalance": [3],
    "YearsAtCompany": [3],
    "YearsInCurrentRole": [2],
    "YearsSinceLastPromotion": [1],
    "YearsWithCurrManager": [2],
    "EmployeeCount": [1],
    "EmployeeNumber": [9999],
    "Over18": ["Y"],
    "StandardHours": [80],
}

employee_df = pd.DataFrame(employee_data)


# -----------------------------
# Make prediction
# -----------------------------
prediction = model.predict(employee_df)[0]

if prediction == 1:
    print("\nPrediction: The employee is likely to leave the company.")
else:
    print("\nPrediction: The employee is likely to stay with the company.")


# -----------------------------
# Prediction probability
# -----------------------------
probabilities = model.predict_proba(employee_df)[0]

print(f"Probability of staying: {probabilities[0]:.2%}")
print(f"Probability of leaving: {probabilities[1]:.2%}")