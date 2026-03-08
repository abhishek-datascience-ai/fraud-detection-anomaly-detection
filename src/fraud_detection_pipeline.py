import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/creditcard.csv")


# =========================
# 2. Data Cleaning
# =========================
df = df.drop_duplicates()
df = df.dropna()


# =========================
# 3. Feature Selection
# =========================
X = df.drop("Class", axis=1)
y = df["Class"]


# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_original = X_test.copy()


# =========================
# 5. Handle Class Imbalance
#    Apply only on training data
# =========================
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# =========================
# 6. Feature Scaling
# =========================
scaler = StandardScaler()

X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


# =========================
# 7. Train XGBoost Model
# =========================
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train_resampled, y_train_resampled)


# =========================
# 8. Predictions
# =========================
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]


# =========================
# 9. Evaluation
# =========================
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =========================
# 10. Prediction Function
# =========================
def predict_transaction(sample):
    sample_df = pd.DataFrame([sample])
    sample_scaled = scaler.transform(sample_df)

    prediction = xgb_model.predict(sample_scaled)[0]
    probability = xgb_model.predict_proba(sample_scaled)[0][1]

    if prediction == 1:
        print("Prediction: Fraud Transaction")
    else:
        print("Prediction: Normal Transaction")

    print("Fraud Probability:", probability)


# =========================
# 11. Sample Prediction
# =========================
print("\nSample Prediction:")
sample_transaction = X_test_original.iloc[0].to_dict()
predict_transaction(sample_transaction)