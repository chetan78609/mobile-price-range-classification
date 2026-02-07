import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("📱 Mobile Price Range Classification App")
st.write("Welcome to Mobile Price Range Classification.")

FEATURE_COLUMNS = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc",
    "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
    "pc", "px_height", "px_width", "ram", "sc_h", "sc_w",
    "talk_time", "three_g", "touch_screen", "wifi"
]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Range Classification",
    layout="wide"
)

# -------------------------------------------------
# Load models
# -------------------------------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

model_name = st.selectbox("Select a model", MODEL_PATHS.keys())
model = joblib.load(MODEL_PATHS[model_name])

st.subheader("⬇️ Download Sample Test Dataset")

with open("D:/BITS Pilani/Semester 2/ML/Assignment II/data/test.csv", "rb") as file:
    st.download_button(
        label="Download test_data.csv",
        data=file,
        file_name="test.csv",
        mime="text/csv"
    )


# -------------------------------------------------
# File upload (Prediction only)
# -------------------------------------------------
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (price_range is NOT required)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Dataset Preview", df.head())

    # -------------------------------
    # Separate target if present
    # -------------------------------
    if "price_range" in df.columns:
        y_true = df["price_range"]
        X = df.drop(columns=["price_range"])
        evaluation_mode = True
    else:
        X = df.copy()
        y_true = None
        evaluation_mode = False

    # -------------------------------
    # Drop extra columns
    # -------------------------------
    extra_cols = set(X.columns) - set(FEATURE_COLUMNS)
    if extra_cols:
        st.warning(f"Dropping extra columns: {list(extra_cols)}")
        X = X.drop(columns=list(extra_cols))

    # -------------------------------
    # Check for missing required columns
    # -------------------------------
    missing_cols = set(FEATURE_COLUMNS) - set(X.columns)
    if missing_cols:
        st.error(f"Missing required columns: {list(missing_cols)}")
        st.stop()

    # -------------------------------
    # Ensure correct column order
    # -------------------------------
    X = X[FEATURE_COLUMNS]

    # -------------------------------
    # Predictions (Pipeline handles scaling)
    # -------------------------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # -------------------------------------------------
    # Evaluation Metrics (ONLY if price_range exists)
    # -------------------------------------------------
    if evaluation_mode:
        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 3))
        col1.metric("Precision", round(precision_score(y_true, y_pred, average="weighted"), 3))

        col2.metric("Recall", round(recall_score(y_true, y_pred, average="weighted"), 3))
        col2.metric("F1 Score", round(f1_score(y_true, y_pred, average="weighted"), 3))

        col3.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 3))
        col3.metric(
            "AUC",
            round(roc_auc_score(y_true, y_prob, multi_class="ovr"), 3)
        )

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

    # -------------------------------
    # Display predictions
    # -------------------------------
    st.subheader("🔮 Predictions on Uploaded Data")

    pred_df = X.copy()
    pred_df["Predicted_price_range"] = y_pred

    st.dataframe(pred_df.head(10))
