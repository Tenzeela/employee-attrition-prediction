# Employee Attrition Prediction

This project predicts whether an employee is likely to leave a company using machine learning classification models.

The project includes data analysis, preprocessing, model training, model evaluation, hyperparameter tuning, and prediction for new employee records.

## Project Objectives

* Analyze employee attrition patterns.
* Identify factors associated with employee turnover.
* Train multiple machine learning classification models.
* Handle class imbalance in the target variable.
* Tune the Support Vector Machine model.
* Predict employee attrition for new employee data.

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains information about employees, including:

* Age
* Department
* Job Role
* Job Satisfaction
* Monthly Income
* Overtime
* Work-Life Balance
* Years at Company
* Total Working Years
* Attrition

The target variable is `Attrition`:

* `Yes`: The employee left the company.
* `No`: The employee stayed with the company.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Git
* GitHub

## Machine Learning Models

The following machine learning models were trained and evaluated:

1. Support Vector Machine
2. Gradient Boosting Classifier
3. XGBoost Classifier

The Support Vector Machine model was selected for further tuning because it provided the strongest attrition-detection performance after class balancing was applied.

## Project Workflow

1. Load the dataset.
2. Perform exploratory data analysis.
3. Check missing values and duplicate records.
4. Analyze employee attrition patterns.
5. Separate the features and target variable.
6. Remove irrelevant columns.
7. Encode categorical features.
8. Impute missing values where required.
9. Scale numerical features.
10. Split the dataset into training and testing sets.
11. Train multiple classification models.
12. Evaluate the trained models.
13. Handle class imbalance.
14. Tune the SVM model using `GridSearchCV`.
15. Save the final trained model.
16. Use the saved model to predict attrition for a new employee record.

## Handling Class Imbalance

The dataset contains more employees who stayed with the company than employees who left. This creates an imbalance in the target variable.

To improve the detection of employees who may leave, the tuned SVM model uses the `class_weight="balanced"` parameter:

```python
SVC(
    class_weight="balanced",
    probability=True
)
```

The `class_weight="balanced"` parameter gives greater importance to the minority class. In this project, the minority class represents employees who left the company.

This helps the model identify more potential attrition cases instead of focusing mainly on employees who stayed.

## Model Evaluation Metrics

The models were evaluated using the following metrics:

* **Accuracy:** The overall proportion of correct predictions.
* **Precision:** The proportion of predicted attrition cases that were actually attrition cases.
* **Recall:** The proportion of actual attrition cases correctly identified by the model.
* **F1-score:** The harmonic mean of precision and recall.
* **Confusion Matrix:** A table showing correct and incorrect predictions for each class.

Recall is particularly important in this project because failing to identify an employee who may leave could prevent an organization from taking early retention measures.

## Model Comparison

The following results were obtained on the test dataset:

| Model                        | Accuracy | Precision | Recall | F1-score |
| ---------------------------- | -------: | --------: | -----: | -------: |
| Support Vector Machine       |   86.05% |    75.00% | 19.15% |   30.51% |
| Gradient Boosting Classifier |   84.69% |    60.00% | 12.77% |   21.05% |
| XGBoost Classifier           |   85.71% |    72.73% | 17.02% |   27.59% |
| Tuned SVM                    |   83.33% |    48.33% | 61.70% |   54.21% |

The original models achieved higher accuracy, but their recall for the attrition class was relatively low.

The tuned SVM achieved a lower overall accuracy but substantially improved recall for employees who left the company. This makes the tuned SVM more useful when detecting potential attrition is more important than maximizing overall accuracy.

## Tuned SVM Results

The tuned SVM model achieved the following results on the test dataset:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 83.33% |
| Precision | 48.33% |
| Recall    | 61.70% |
| F1-score  | 54.21% |

The tuned SVM achieved a recall of approximately **61.70%** for employees who left the company.

This means that the model correctly identified approximately 61.70% of the employees in the test dataset who actually left the company.

## Hyperparameter Tuning

The SVM model was tuned using `GridSearchCV`.

The best-performing hyperparameter combination was:

| Parameter      | Selected Value |
| -------------- | -------------- |
| `C`            | `1`            |
| `class_weight` | `balanced`     |
| `gamma`        | `scale`        |
| `kernel`       | `rbf`          |

The best cross-validation F1-score was approximately **0.568**.

The use of `class_weight="balanced"` helped improve the model's ability to detect the minority attrition class.

## Model Prediction

The trained model was saved using Joblib:

```python
joblib.dump(model, "results/final_attrition_model.pkl")
```

The saved model can be loaded and used to predict attrition for a new employee record:

```python
import joblib

model = joblib.load("results/final_attrition_model.pkl")

prediction = model.predict(new_employee_data)
prediction_probability = model.predict_proba(new_employee_data)
```

The prediction script provides the predicted class and the estimated probability of employee attrition.

Example prediction output:

```text
Prediction: Employee is likely to stay

Probability of staying: 81.98%
Probability of leaving: 18.02%
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/employee-attrition-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd employee-attrition-prediction
```

### 3. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Data Analysis

```bash
python src/data_analysis.py
```

### Train the Models

```bash
python src/train_model.py
```

### Tune the SVM Model

```bash
python src/tune_model.py
```

### Make a Prediction

```bash
python src/predict.py
```

## Project Structure

```text
employee-attrition-prediction/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── results/
│   ├── attrition_by_job_satisfaction.png
│   ├── attrition_by_overtime.png
│   ├── attrition_distribution.png
│   ├── monthly_income_distribution.png
│   ├── confusion_matrix_gradient_boosting_classifier.png
│   ├── confusion_matrix_support_vector_machine.png
│   ├── confusion_matrix_tuned_svm.png
│   ├── confusion_matrix_xgboost_classifier.png
│   ├── final_attrition_model.pkl
│   ├── model_evaluation_results.csv
│   ├── svm_tuning_results.csv
│   └── tuned_svm_evaluation_results.csv
│
└── src/
    ├── data_analysis.py
    ├── train_model.py
    ├── tune_model.py
    └── predict.py
```

## Results and Findings

The project demonstrated that:

* Employee attrition can be predicted using classification algorithms.
* Class imbalance can significantly affect attrition detection.
* Accuracy alone is not sufficient for evaluating employee attrition models.
* The original models achieved higher accuracy but lower attrition recall.
* The tuned SVM improved recall for the attrition class.
* Precision, recall, and F1-score are important when evaluating employee turnover prediction.
* Overtime, job satisfaction, income, and other employee-related factors can be explored to understand attrition patterns.

## Limitations

* The dataset is based on historical employee information and may not represent every organization.
* Model predictions may contain false positives and false negatives.
* The model does not establish that a particular factor directly causes employee attrition.
* The model's performance may change when applied to a different company or dataset.
* Predictions should be interpreted as estimates rather than guaranteed outcomes.

## Responsible Use

This project is intended for educational and analytical purposes.

Employee attrition predictions should not be used as the sole basis for employment decisions, termination, promotion, compensation, or other actions affecting employees.

Any real-world use should include appropriate privacy protections, fairness evaluation, human oversight, and review for potential bias.

## Future Improvements

Possible future improvements include:

* Testing additional classification algorithms.
* Applying advanced class-imbalance techniques such as SMOTE.
* Performing more extensive hyperparameter tuning.
* Evaluating the model using stratified cross-validation.
* Improving feature selection.
* Calibrating prediction probabilities.
* Adding an interactive dashboard.
* Monitoring model performance on new employee data.
* Performing fairness and bias analysis.
* Deploying the model through an API or web application.

## Author

**Tenzeela Saeed**

This project was developed as part of practical machine learning and data science training.
