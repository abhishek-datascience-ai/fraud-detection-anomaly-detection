# Fraud Detection (Anomaly Detection)

End-to-end machine learning project for detecting fraudulent credit card transactions using **Isolation Forest**, **XGBoost**, **SMOTE**, and **Streamlit**.

## Overview

This project solves a highly imbalanced fraud detection problem using both:

- **Anomaly Detection** with Isolation Forest
- **Supervised Classification** with XGBoost

The workflow includes data cleaning, EDA, imbalance handling, feature scaling, model training, evaluation, and dashboard development.

## Problem Statement

Fraud transactions are very rare compared to normal transactions, which makes prediction difficult.  
The goal of this project is to identify fraudulent transactions with strong recall and precision while keeping the pipeline simple, practical, and portfolio-ready.

## Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

- Dataset: `creditcard.csv`
- Source: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

### Important Note
The dataset is **not included in this repository** due to GitHub file size / repository limit considerations.  
Please download it manually from Kaggle and place it in:

```text
data/creditcard.csv
```

## Dataset Columns

- `Time`
- `V1` to `V28` → PCA-transformed anonymized features
- `Amount`
- `Class`

### Target column

- `0` = Normal Transaction
- `1` = Fraud Transaction

## Project Structure

```text
fraud_detection_anomaly_detection/
│
├── data/
│   └── creditcard.csv
├── notebook/
│   └── fraud_detection.ipynb
├── src/
│   └── fraud_detection_pipeline.py
├── dashboard/
│   └── app.py
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Streamlit
- Jupyter Notebook

## Workflow

- Project setup and environment creation
- Dataset collection from Kaggle
- Data loading with Pandas
- Data cleaning and duplicate removal
- Exploratory Data Analysis (EDA)
- Class imbalance handling using SMOTE
- Feature selection
- Feature scaling with StandardScaler
- Isolation Forest model training
- XGBoost model training
- Model evaluation
- Fraud visualization
- Fraud prediction function
- Python pipeline script creation
- Streamlit dashboard development

## Models Used

### Isolation Forest

Used for anomaly detection to identify unusual transaction patterns.

### XGBoost

Used as the main supervised fraud classification model.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Key Features

- End-to-end fraud detection pipeline
- Handles severe class imbalance
- Includes anomaly detection and classification
- Notebook-based experimentation
- Reusable Python pipeline
- Streamlit dashboard for prediction

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/fraud-detection-anomaly-detection.git
cd fraud-detection-anomaly-detection
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

### 1. Run Notebook

```bash
jupyter notebook
```

Open:

```text
notebook/fraud_detection.ipynb
```

### 2. Run Python Pipeline

```bash
python src/fraud_detection_pipeline.py
```

### 3. Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

## Streamlit Dashboard

The dashboard allows the user to:

- enter transaction feature values
- predict fraud / normal transaction
- view fraud probability

## Learning Outcomes

This project demonstrates practical skills in:

- anomaly detection
- fraud analytics
- imbalanced learning
- SMOTE
- Isolation Forest
- XGBoost
- feature scaling
- model evaluation
- Streamlit dashboard development

## Future Improvements

- Save trained model with `joblib` / `pickle`
- Load saved model directly in dashboard
- Add threshold tuning
- Add SHAP explainability
- Deploy dashboard online
- Add batch prediction support

## Author

**Abhishek**

## Note

After downloading the dataset from Kaggle, place it here:

```text
data/creditcard.csv
```

Without this file, the notebook, pipeline, and dashboard will not run.