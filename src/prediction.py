import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS


def prepare_user_input(
    scaler,
    total_4_year_cost,
    graduation_rate,
    typical_years_to_graduate,
    average_loan_amount
):
    input_data = pd.DataFrame({
        "Total 4 Year Cost": [total_4_year_cost],
        "Graduation Rate": [graduation_rate],
        "Typical Years to graduate": [typical_years_to_graduate],
        "Average Loan Amount": [average_loan_amount]
    })

    input_scaled = scaler.transform(input_data[FEATURE_COLUMNS])

    return input_scaled


def predict_roi(model, input_data):
    prediction = model.predict(input_data)[0]
    return prediction