# Data manipulation
import platform
import sys
from datetime import datetime

import openpyxl
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import tf
from displayfunction import display

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Model selection
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

# Machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)

# Model saving
import joblib

import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully!")

import pandas as pd

df = pd.read_csv(
    "/Users/ipsitabiswal/PycharmProjects/PythonProject/ML_Assignment2/WA_Fn-UseC_-HR-Employee-Attrition.xls",
    encoding="utf-8-sig"
)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)

df.head()

# Display first five records
display(df.head())

# Dataset information
df.info()

# Statistical summary
display(df.describe())

# Check dataset dimensions
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Check column names
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
missing_values = df.isnull().sum()

print("Missing values:")
print(missing_values[missing_values > 0])

# Check duplicate records
print(
    "Duplicate records:",
    df.duplicated().sum()
)

print(df["Attrition"].value_counts())

# Visualize target variable
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition",
    hue="Attrition",
    palette="Set2",
    legend=False
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.show()

# Create a copy
data = df.copy()

# Remove identifier and constant columns
data.drop(
    columns=[
        "EmployeeNumber",
        "EmployeeCount",
        "StandardHours",
        "Over18"
    ],
    inplace=True
)

# Convert target into binary format
data["Attrition"] = data["Attrition"].map({
    "Yes": 1,
    "No": 0
})

print("Preprocessing completed!")

print("New dataset shape:", data.shape)

print(data.head())

X = data.drop("Attrition", axis=1)
y = data["Attrition"]

print("Number of instances:", len(data))
print("Number of input features:", X.shape[1])

assert len(data) >= 1000, "Dataset must contain at least 1000 instances"
assert X.shape[1] >= 12, "At least 12 input features are required"

print("\nDataset requirements satisfied!")

plt.figure(figsize=(7, 5))

sns.countplot(
    data=data,
    x="Attrition",
    hue="Attrition",
    palette="viridis",
    legend=False
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=data,
    x="Age",
    hue="Attrition",
    kde=True,
    multiple="stack"
)

plt.title("Age Distribution Based on Attrition")
plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Attrition",
    y="MonthlyIncome",
    hue="Attrition",
    palette="Set2",
    legend=False
)

plt.title("Monthly Income vs Employee Attrition")
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    data=data,
    x="JobSatisfaction",
    hue="Attrition",
    palette="Set1"
)

plt.title("Job Satisfaction vs Attrition")
plt.show()

plt.figure(figsize=(16, 12))

numeric_data = data.select_dtypes(
    include=np.number
)

sns.heatmap(
    numeric_data.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Feature Correlation Heatmap")
plt.show()

# Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'Yes': 1, 'No': 0})

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_features = X.select_dtypes(
    include=['object', 'category']
).columns

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit and transform training data
X_train = preprocessor.fit_transform(X_train)

# Transform testing data
X_test = preprocessor.transform(X_test)

print("Feature preprocessing pipeline created!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)

# Logistic Regression model
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Train model
lr_model.fit(X_train, y_train)

# Predictions
lr_pred = lr_model.predict(X_test)

# Probability predictions
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Results")
print(classification_report(y_test, lr_pred))

# Decision Tree model
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

# Train model
dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)

# Probability predictions
dt_prob = dt_model.predict_proba(X_test)[:, 1]

print("Decision Tree Results")
print(classification_report(y_test, dt_pred))


# KNN model
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

# Train model
knn_model.fit(X_train, y_train)

# Predictions
knn_pred = knn_model.predict(X_test)

# Probability predictions
knn_prob = knn_model.predict_proba(X_test)[:, 1]

print("KNN Results")
print(classification_report(y_test, knn_pred))

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Probability predictions
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print(classification_report(y_test, rf_pred))

# Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train model
gb_model.fit(X_train, y_train)

# Predictions
gb_pred = gb_model.predict(X_test)

# Probability predictions
gb_prob = gb_model.predict_proba(X_test)[:, 1]

print("Gradient Boosting Results")
print(classification_report(y_test, gb_pred))

# Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train model
nb_model.fit(X_train, y_train)

# Predictions
nb_pred = nb_model.predict(X_test)

# Probability predictions
nb_prob = nb_model.predict_proba(X_test)[:, 1]

print("Gaussian Naive Bayes Results")
print(classification_report(y_test, nb_pred))



print("Five machine learning models created!")

# Store model predictions
models = {
    'Logistic Regression': (lr_pred, lr_prob),
    'Decision Tree': (dt_pred, dt_prob),
    'KNN': (knn_pred, knn_prob),
    'Random Forest': (rf_pred, rf_prob),
    'Gradient Boosting': (gb_pred, gb_prob),
    'Gaussian Naive Bayes': (nb_pred, nb_prob)
}

# Evaluate all models
results = []

for name, (pred, prob) in models.items():

    results.append({
        'Model': name,

        'Accuracy': accuracy_score(y_test, pred),

        'Precision': precision_score(
            y_test, pred, zero_division=0
        ),

        'Recall': recall_score(
            y_test, pred, zero_division=0
        ),

        'F1 Score': f1_score(
            y_test, pred, zero_division=0
        ),

        'ROC-AUC': roc_auc_score(y_test, prob),

        'MCC Score': matthews_corrcoef(y_test, pred)
    })

# Create comparison DataFrame
results_df = pd.DataFrame(results)

# Display results
print("\nEmployee Attrition Model Comparison")
print(results_df.to_string(index=False))

metrics = [
    'Accuracy',
    'Precision',
    'Recall',
    'F1 Score',
    'ROC-AUC',
    'MCC Score'
]

results_melted = results_df.melt(
    id_vars='Model',
    value_vars=metrics,
    var_name='Metric',
    value_name='Score'
)

plt.figure(figsize=(14, 7))

sns.barplot(
    data=results_melted,
    x='Model',
    y='Score',
    hue='Metric'
)

plt.title('Employee Attrition - Performance Comparison')
plt.xlabel('Machine Learning Models')
plt.ylabel('Performance Score')

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

results_df = results_df.sort_values(
    by='MCC Score',
    ascending=False
)

print("\nModel Ranking Based on MCC Score")
print(results_df[['Model', 'MCC Score']])

best_model = results_df.iloc[0]['Model']

print("\nBest Model based on MCC:", best_model)

# include the screen shot here
print("System Environment Information:")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version}")
print(f"OS Platform: {sys.platform}")
print(f"OS Architecture: {platform.architecture()[0]}")
print(f"Processor: {platform.processor()}")

