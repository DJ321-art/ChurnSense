import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def clean_data(df):
    df = df.copy()

    df = df.drop(columns=["customerID"], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    return df

def add_engineered_features(df):
    df = df.copy()

    df["NumServices"] = df[SERVICE_COLS].apply(
        lambda row: (row == "Yes").sum(),
        axis=1
    )

    return df

def encode_target(df):
    df = df.copy()

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({
            "No": 0,
            "Yes": 1
        })

    return df

def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["str"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor

