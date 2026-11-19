# ML_Assignment2
Python Code for Machine Learning

Employee Attrition Prediction Using Machine Learning
1. Project Overview

Employee Attrition Prediction is a Machine Learning project that aims to predict whether an employee is likely to leave an organization based on various employee-related factors.

Employee attrition is a major challenge for organizations because it affects productivity, recruitment costs, employee retention and overall organizational performance.

This project implements multiple Machine Learning classification algorithms to predict employee attrition and compare their performance using different evaluation metrics.

2. Objectives
Predict whether an employee will leave the organization.
Apply different Machine Learning classification algorithms.
Perform data preprocessing and feature engineering.
Identify important factors influencing employee attrition.
Compare the performance of different classification models.
Identify the best-performing model using evaluation metrics.
3. Dataset Description

Dataset: IBM HR Analytics Employee Attrition & Performance

File Name: WA_Fn-UseC_-HR-Employee-Attrition.csv

The dataset contains employee information such as:

Age
Gender
Department
Education
Job Role
Monthly Income
Job Satisfaction
Work-Life Balance
Years at Company
Years in Current Role
Business Travel
OverTime
Distance From Home
Environment Satisfaction

Target Variable: Attrition

Yes: Employee leaves the organization.
No: Employee stays in the organization.
4. Machine Learning Algorithms Used

This project implements the following Machine Learning algorithms:

4.1 Logistic Regression

Logistic Regression is a supervised classification algorithm that predicts the probability of employee attrition.

4.2 Decision Tree

Decision Tree is a tree-based classification algorithm that makes predictions using decision rules based on employee characteristics.

4.3 K-Nearest Neighbors (KNN)

KNN classifies employees based on the similarity between their characteristics and those of other employees in the training dataset.

4.4 Ensemble Learning

Two ensemble learning algorithms are implemented:

Random Forest: Combines multiple decision trees to improve prediction performance.
Gradient Boosting: Builds sequential decision trees to reduce prediction errors.
4.5 Bayesian Learning

Gaussian Naive Bayes uses Bayes' theorem to calculate the probability of employee attrition based on employee-related features.

5. Data Preprocessing

The following preprocessing techniques are applied:

Handling missing values using imputation.
Separating numerical and categorical features.
Encoding categorical variables using One-Hot Encoding.
Scaling numerical features using StandardScaler.
Splitting the dataset into training and testing datasets.
Handling class imbalance using class weighting where supported.

The dataset is divided into:

Training Data: 80%
Testing Data: 20%
6. Model Evaluation Metrics

The performance of the classification algorithms is evaluated using the following metrics:

Metric	Description
Accuracy	Measures the proportion of correct predictions.
Precision	Measures how many predicted attrition cases are actually attrition cases.
Recall	Measures how many actual attrition cases are correctly identified.
F1 Score	Harmonic mean of precision and recall.
ROC-AUC	Measures the model's ability to distinguish between classes.
MCC Score	Measures the correlation between actual and predicted classifications.
Matthews Correlation Coefficient (MCC)

MCC is particularly useful for imbalanced classification datasets.

Its values range from -1 to +1:

+1: Perfect classification.
0: No correlation between actual and predicted classifications.
-1: Complete disagreement between actual and predicted classifications.
7. Technologies and Libraries Used
Python 3
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
PyCharm
8. Project Structure
text
Employee_Attrition_Prediction/
│
├── Employee_Attrition_Prediction.py
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── README.md
│
└── requirements.txt

9. Installation and Setup
Step 1: Clone the Repository
bash
git clone <your-github-repository-url>

Step 2: Navigate to the Project Directory
bash
cd Employee_Attrition_Prediction

Step 3: Install Required Libraries
bash
pip install -r requirements.txt


Alternatively, install the dependencies individually:

bash
pip install pandas numpy matplotlib seaborn scikit-learn

Step 4: Run the Python Program
bash
python Employee_Attrition_Prediction.py

10. Expected Outputs

After executing the program, the following outputs are generated:

Dataset information and preprocessing results.
Training and testing dataset dimensions.
Classification reports for all implemented models.
Accuracy, Precision, Recall and F1 Score comparisons.
ROC-AUC and MCC Score comparisons.
Confusion matrices.
ROC curve comparison.
Feature importance visualization.
Best-performing model based on the selected evaluation metric.
11. Results and Model Comparison

The performance of all implemented algorithms is compared using the evaluation metrics.

Algorithm	Accuracy	Precision	Recall	F1 Score	ROC-AUC	MCC
Logistic Regression	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated
Decision Tree	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated
KNN	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated
Random Forest	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated
Gradient Boosting	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated
Gaussian Naive Bayes	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated	To be calculated

Note: Actual performance values will be populated after executing the Python program.

12. Conclusion

This project demonstrates the application of different Machine Learning classification techniques for predicting employee attrition.

By comparing Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting and Gaussian Naive Bayes, the project evaluates how different algorithms perform on employee attrition data.

The evaluation metrics, particularly Recall, F1 Score, ROC-AUC and Matthews Correlation Coefficient, help identify models that can effectively recognize employees who may leave the organization.

The results can help organizations understand employee attrition patterns and develop better employee retention strategies.

13. Future Enhancements
Apply hyperparameter tuning to improve model performance.
Implement SMOTE to address class imbalance.
Use SHAP for model explainability.
Develop an interactive employee attrition prediction web application.
Deploy the best-performing model using Flask or Streamlit.
14. Author

Project: Employee Attrition Prediction Using Machine Learning

Domain: Human Resource Analytics

Type: Supervised Machine Learning - Binary Classification