
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------
# Paths
# ---------------------------------
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET LOADED ==========")
print("Dataset shape:", df.shape)


# ---------------------------------
# Remove unnecessary columns
# ---------------------------------
columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop, errors="ignore")


# ---------------------------------
# Encode target variable
# ---------------------------------
df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# ---------------------------------
# Identify column types
# ---------------------------------
numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns


# ---------------------------------
# Preprocessing pipelines
# ---------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ---------------------------------
# Build SVM pipeline
# ---------------------------------
svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            SVC(
                probability=True,
                random_state=42
            )
        )
    ]
)


# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------
# Hyperparameter grid
# ---------------------------------
param_grid = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", "auto"],
    "model__kernel": ["rbf"],
    "model__class_weight": ["balanced"]
}


# ---------------------------------
# Cross-validation configuration
# ---------------------------------
cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ---------------------------------
# Grid search
# ---------------------------------
grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

print("\n========== STARTING GRID SEARCH ==========")

grid_search.fit(X_train, y_train)


# ---------------------------------
# Best model information
# ---------------------------------
print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

print("\nBest cross-validation F1-score:", round(grid_search.best_score_, 4))


# ---------------------------------
# Evaluate tuned model
# ---------------------------------
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n========== TUNED SVM RESULTS ==========")
print("Accuracy: ", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:   ", round(recall, 4))
print("F1-score: ", round(f1, 4))

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Attrition", "Attrition"],
        zero_division=0
    )
)


# ---------------------------------
# Save confusion matrix
# ---------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Attrition", "Attrition"],
    yticklabels=["No Attrition", "Attrition"]
)

plt.title("Confusion Matrix - Tuned SVM")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix_tuned_svm.png"),
    dpi=300
)

plt.close()


# ---------------------------------
# Save tuning results
# ---------------------------------
tuning_results = pd.DataFrame(grid_search.cv_results_)

tuning_results = tuning_results[
    [
        "param_model__C",
        "param_model__gamma",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
]

tuning_results.to_csv(
    os.path.join(RESULTS_DIR, "svm_tuning_results.csv"),
    index=False
)


# ---------------------------------
# Save final tuned model metrics
# ---------------------------------
final_results = pd.DataFrame(
    [
        {
            "Model": "Tuned Support Vector Machine",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }
    ]
)

final_results.to_csv(
    os.path.join(RESULTS_DIR, "tuned_svm_evaluation_results.csv"),
    index=False
)


# ---------------------------------
# Save final tuned model
# ---------------------------------
MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "final_attrition_model.pkl"
)

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nFinal tuned model saved to:", MODEL_PATH)

print("\n========== TUNING COMPLETED ==========")
print("Best model parameters saved in the terminal output.")
print("Confusion matrix saved in results/")
print("Tuning results saved in results/svm_tuning_results.csv")
print("Evaluation metrics saved in results/tuned_svm_evaluation_results.csv")
print("Final model saved in results/final_attrition_model.pkl")