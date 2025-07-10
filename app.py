import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('model.pkl')

st.title("House Price Prediction App")

st.divider()

st.write("This app predicts the price of a house using Machine Learning. For using this app, please fill the inputs in the UI and click the predict button:")

st.divider()

bedrooms = st.number_input("Number of bedrooms", min_value=1, value=1)
bathrooms = st.number_input("Number of bathrooms", min_value=1, value=1)
living_area = st.number_input("Living area (in square feet)", min_value=1000, value=1500)
condition = st.number_input("Condition of the house", min_value=1, value=3)
schools_nearby = st.number_input("Number of schools nearby", min_value=0, value=1)

st.divider()

input_df = pd.DataFrame([{
    'number of bedrooms': bedrooms,
    'number of bathrooms': bathrooms,
    'living area': living_area,
    'condition of the house': condition,
    'Number of schools nearby': schools_nearby
}])

prediction = model.predict(input_df)


predict_button = st.button("Predict")

if predict_button:
    st.balloons()
    prediction = model.predict(input_df)
    st.write(f"The predicted price of the house is: {prediction[0]:,.2f}")
