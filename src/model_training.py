# ===============================
# Customer Churn Model Training
# ===============================

# Import Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import data_preprocessing


# ===============================
# Load and Prepare Data
# ===============================

def prepare_data():

    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = data_preprocessing.load_data(data_path)
    df = data_preprocessing.clean_data(df)
    df = data_preprocessing.encode_target(df)
    df = data_preprocessing.encode_features(df)

    X_train, X_test, y_train, y_test = data_preprocessing.split_data(df)

    return X_train, X_test, y_train, y_test


# ===============================
# Train Model
# ===============================

def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


# ===============================
# Evaluate Model
# ===============================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(" Model Training Completed")
    print("Model Accuracy:", accuracy)


# ===============================
# Main Function
# ===============================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
