# ===============================
# Customer Churn Data Preprocessing
# ===============================

# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ===============================
# Load Dataset
# ===============================
def load_data(path):
    df = pd.read_csv(path)
    return df


# ===============================
# Data Cleaning
# ===============================
def clean_data(df):

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove missing values
    df.dropna(inplace=True)

    return df


# ===============================
# Encode Target Variable
# ===============================
def encode_target(df):

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# ===============================
# Encode Categorical Features
# ===============================
def encode_features(df):

    le = LabelEncoder()

    for col in df.select_dtypes(include=["object"]).columns:
        if col != "customerID":
            df[col] = le.fit_transform(df[col])

    return df


# ===============================
# Train Test Split
# ===============================
def split_data(df):

    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ===============================
# Main Function
# ===============================
if __name__ == "__main__":

    # Dataset Path
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = load_data(data_path)

    df = clean_data(df)

    df = encode_target(df)

    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Data Preprocessing Completed")
    print("Training Data Shape:", X_train.shape)

