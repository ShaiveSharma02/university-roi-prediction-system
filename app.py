import streamlit as st

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_data,
    preprocess_data,
    split_data
)
from src.model_training import train_model, evaluate_model
from src.prediction import prepare_user_input, predict_roi
from src.visualization import plot_feature_importance


DATA_PATH = "data/datasetROI.csv"


@st.cache_resource
def build_model():
    data = load_data(DATA_PATH)

    X, y, scaler = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    mae, r2 = evaluate_model(model, X_test, y_test)

    return data, model, scaler, mae, r2


st.set_page_config(
    page_title="University ROI Prediction System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University ROI Prediction System")

st.write(
    "A machine learning web application that predicts the estimated 20-year net ROI "
    "of a university based on cost, graduation rate, graduation timeline, and loan amount."
)

data, model, scaler, mae, r2 = build_model()

st.sidebar.header("University Financial Details")

total_4_year_cost = st.sidebar.number_input(
    "Total 4 Year Cost ($)",
    min_value=0,
    value=180000,
    step=5000
)

graduation_rate = st.sidebar.slider(
    "Graduation Rate (%)",
    min_value=0,
    max_value=100,
    value=75
)

typical_years_to_graduate = st.sidebar.number_input(
    "Typical Years to Graduate",
    min_value=1.0,
    max_value=8.0,
    value=4.0,
    step=0.1
)

average_loan_amount = st.sidebar.number_input(
    "Average Loan Amount ($)",
    min_value=0,
    value=25000,
    step=1000
)

if st.sidebar.button("Predict ROI"):

    input_data = prepare_user_input(
        scaler,
        total_4_year_cost,
        graduation_rate,
        typical_years_to_graduate,
        average_loan_amount
    )

    predicted_roi = predict_roi(model, input_data)

    st.subheader("Prediction Result")

    st.success(
        f"Estimated 20-Year Net ROI: ${predicted_roi:,.2f}"
    )

st.divider()

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Absolute Error", f"${mae:,.2f}")

with col2:
    st.metric("R² Score", f"{r2:.2f}")

st.subheader("Feature Importance")

fig = plot_feature_importance(model, FEATURE_COLUMNS)
st.pyplot(fig)

st.subheader("Dataset Preview")

st.dataframe(data.head(10))