import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("admission.csv")   # your CSV name
    return df

# ---------- Train model ----------
@st.cache_resource
def train_model(df):
    X = df[['maths', 'science', 'english', 'total', 'avg', 'GRE_score']]
    y = df['Chance_of_Admit']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return model, mae, mse

# ---------- Streamlit UI ----------
def main():
    st.title("Student Admission Prediction")

    # 1) Show dataset
    df = load_data()
    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Shape of data")
    st.write(df.shape)

    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Missing values")
    st.write(df.isna().sum())

    # 2) Train model and show errors
    model, mae, mse = train_model(df)
    st.subheader("Model Performance")
    st.write(f"Mean Absolute Error: {mae:.3f}")
    st.write(f"Mean Squared Error: {mse:.3f}")

    # 3) User input for prediction
    st.subheader("Try Your Own Marks")

    maths = st.number_input("Maths", 0, 100, 80)
    science = st.number_input("Science", 0, 100, 75)
    english = st.number_input("English", 0, 100, 78)
    total = maths + science + english
    avg = total / 3.0
    gre = st.number_input("GRE Score", 250, 340, 310)

    st.write(f"Calculated total: {total}")
    st.write(f"Calculated average: {avg:.2f}")

    if st.button("Predict Chance of Admit"):
        X_new = np.array([[maths, science, english, total, avg, gre]])
        pred = model.predict(X_new)[0]
        st.success(f"Predicted Chance of Admit: {pred:.2f}")

if __name__ == "__main__":
    main()
