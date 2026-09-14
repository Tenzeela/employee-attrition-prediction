import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------
# File paths
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
# Yes = 1, No = 0
# ---------------------------------
df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# ---------------------------------
# Identify column types
# ---------------------------------
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ---------------------------------
# Preprocessing pipelines
# ---------------------------------
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", __import__(
            "sklearn.preprocessing",
            fromlist=["OneHotEncoder"]
        ).OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ]
)


# ---------------------------------
# Train-test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ---------------------------------
# Define models
# ---------------------------------
models = {
    "Support Vector Machine": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        probability=True,
        random_state=42
    ),

    "Gradient Boosting Classifier": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost Classifier": XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ---------------------------------
# Train and evaluate models
# ---------------------------------
evaluation_results = []

for model_name, model in models.items():

    print(f"\n========== {model_name} ==========")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Attrition", "Attrition"],
            zero_division=0
        )
    )

    # Save metrics
    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    # ---------------------------------
    # Confusion matrix
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

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    safe_model_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"confusion_matrix_{safe_model_name}.png"
        ),
        dpi=300
    )

    plt.close()


# ---------------------------------
# Save evaluation results
# ---------------------------------
results_df = pd.DataFrame(evaluation_results)

results_df.to_csv(
    os.path.join(RESULTS_DIR, "model_evaluation_results.csv"),
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df)

print("\n========== TRAINING COMPLETED ==========")
print("Evaluation results and confusion matrices saved in results/.")