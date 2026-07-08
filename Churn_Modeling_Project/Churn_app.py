import streamlit as st
import pickle
import numpy as np

#Load Model
with open('churn_model.pkl','rb') as f:
    model=pickle.load(f)

#Heading of the Webpage
st.title("Customer Churn Prediction")
#Subheading of the Webpage
st.write("Enter the Customer Details below : ")

# User manually enters the values
CreditScore = st.number_input("Credit Score", min_value=0, max_value=1000, value=500)
Geography = st.number_input("Geography (0 = France, 1 = Germany, 2 = Spain)",min_value=0, max_value=2, value=0)
Gender = st.number_input("Gender (0 = Female, 1 = Male)",min_value=0, max_value=1, value=0)
Age = st.number_input("Age", min_value=18, max_value=100, value=41)
Tenure = st.number_input( "Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.number_input("Has Credit Card (0 = No, 1 = Yes)",min_value=0, max_value=1, value=1)
IsActiveMember = st.number_input("Is Active Member (0 = No, 1 = Yes)",min_value=0, max_value=1, value=1)
EstimatedSalary = st.number_input("Estimated Salary",min_value=0.0,max_value=200000.0,value=50000.0)


features=np.array([[CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]])

if st.button("Predict"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is likely to stay.")
