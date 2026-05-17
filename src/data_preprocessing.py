import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Total 4 Year Cost",
    "Graduation Rate",
    "Typical Years to graduate",
    "Average Loan Amount"
]

TARGET_COLUMN = "20 Year Net ROI"


def clean_currency_column(series):
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )


def clean_percentage_column(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(float)
    )


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    return data


def preprocess_data(data):
    data = data.copy()

    for column in [
        "20 Year Net ROI",
        "Total 4 Year Cost",
        "Average Loan Amount"
    ]:
        data[column] = clean_currency_column(data[column])

    data["Graduation Rate"] = clean_percentage_column(
        data["Graduation Rate"]
    )

    data["Typical Years to graduate"] = pd.to_numeric(
        data["Typical Years to graduate"],
        errors="coerce"
    )

    data = data.dropna()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )