import streamlit as st
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# -----------------------------
# Load and train model
# -----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("data/creditcard.csv")

    # cleaning
    df = df.drop_duplicates()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # balance data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # remove Time
    X_resampled = X_resampled.drop("Time", axis=1)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled
    )

    # model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model, scaler, X_resampled.columns


model, scaler, feature_columns = load_model()


# -----------------------------
# App title
# -----------------------------
st.title("Fraud Detection Dashboard")
st.write("Enter transaction feature values to predict fraud.")

st.subheader("Transaction Input")


# -----------------------------
# Input fields
# -----------------------------
user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(f"Enter {col}", value=0.0)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Normal Transaction")

    st.write(f"Fraud Probability: {probability:.4f}")