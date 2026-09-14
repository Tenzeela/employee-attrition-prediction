import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# File paths
# ---------------------------------
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
RESULTS_DIR = "results"


# ---------------------------------
# Create results directory
# ---------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Attrition"].value_counts())

print("\n========== TARGET PERCENTAGES ==========")
print(df["Attrition"].value_counts(normalize=True) * 100)


# ---------------------------------
# Plot 1: Attrition distribution
# ---------------------------------
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Attrition")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "attrition_distribution.png"),
    dpi=300
)

plt.show()


# ---------------------------------
# Plot 2: Attrition by overtime
# ---------------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("Employee Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "attrition_by_overtime.png"),
    dpi=300
)

plt.show()


# ---------------------------------
# Plot 3: Attrition by job satisfaction
# ---------------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title("Employee Attrition by Job Satisfaction")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "attrition_by_job_satisfaction.png"),
    dpi=300
)

plt.show()


# ---------------------------------
# Plot 4: Monthly income distribution
# ---------------------------------
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyIncome",
    hue="Attrition",
    kde=True,
    bins=30
)

plt.title("Monthly Income Distribution by Attrition")
plt.xlabel("Monthly Income")
plt.ylabel("Number of Employees")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "monthly_income_distribution.png"),
    dpi=300
)

plt.show()


print("\n========== ANALYSIS COMPLETED ==========")
print("Plots saved in the results folder.")