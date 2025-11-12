# Traffic Accident Severity Prediction Model

A machine learning project that predicts traffic accident severity using various classification algorithms. This academic project was developed as part of the Research and Innovation module at Botswana Accountancy College.

## 📊 Project Overview

- **Objective**: Develop predictive models to classify traffic accident severity into Fatal, Serious, and Slight injury categories
- **Dataset**: 12,304 traffic accident records with 25+ features including time, location, driver demographics, and road conditions
- **Key Achievement**: XGBoost model achieved **91.2% accuracy** in severity prediction

## 🛠️ Technologies Used

- **Programming**: Python
- **Libraries**: Pandas, Scikit-learn, XGBoost, Joblib
- **Machine Learning**: Logistic Regression, Gradient Boosting, XGBoost
- **Methodology**: CRISP-DM
- **Tools**: Jupyter Notebook, Microsoft Excel (for data visualization)

## 📁 Project Structure
traffic-accident-prediction-model/
├── v5.py # Main Python script for model training
├── project_report.pdf # Detailed project report
└── README.md # This file


📈 Model Performance
Hold-out Test Set (80/20 Split)
Model	Accuracy	Precision	Recall	F1-Score
XGBoost	0.912	0.919	0.912	0.896
Gradient Boosting	0.852	0.859	0.852	0.791
Logistic Regression	0.492	0.780	0.492	0.583
Key Findings
XGBoost significantly outperformed other models

Best at predicting "Slight Injury" cases (F1-score: 0.951)

Maintained good performance on minority classes despite class imbalance

🔧 Features Engineering
Extracted "Hour" feature from time data

Handled missing values with median (numeric) and mode (categorical)

One-hot encoding for categorical variables

Standard scaling for numerical features

Stratified train-test split to preserve class distribution

👥 Team
This project was collaboratively developed by BIDA students at Botswana Accountancy College.

📄 Report
The detailed project report (project_report.pdf) includes:

Comprehensive literature review

Exploratory Data Analysis (EDA) findings

Methodology using CRISP-DM framework

Complete results and analysis

Business recommendations and policy implications
