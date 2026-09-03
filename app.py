import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import streamlit as st
import pandas as pd


###### Loading the model #####
model =tf.keras.models.load_model('artifacts/Ann_model.h5')

###### Loading preprocessor #####
preprocessor = joblib.load("artifacts/churn_model_pipeline.pkl")

###### Streamlit app #####
st.title("ANN Model Churn Prediction")

##### User input #####

columns_name=np.load("artifacts\dataset_splits.npz")['columns']

CreditScore = st.number_input('CreditScore')
Geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
Age = st.slider('Age', 18, 100, 30)
Tenure = st.slider('Tenure', 0, 10, 5)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('NumOfProducts', 1, 8, 1)
HasCrCard = st.selectbox('HasCrCard', [0, 1])
IsActiveMember = st.selectbox('IsActiveMember', [0, 1])
EstimatedSalary = st.number_input('EstimatedSalary')


##### Derived features #####
BalanceToSalaryRatio = Balance / (EstimatedSalary + 1e-6)  
IsHighBalance = 1 if Balance > 97198.540 else 0
IsMultiProductCustomer = 1 if NumOfProducts > 1 else 0

AgeGroup = '18-25' if Age < 26 else '26-35' if Age < 36 else '36-45' if Age < 46 else '46-55' if Age < 56 else '56-65' if Age < 66 else '66+'

CreditScoreBand = 'Low' if CreditScore < 400 else 'Fair' if CreditScore < 600 else 'Good' if CreditScore < 700 else 'Very Good' if CreditScore < 800 else 'Excellent'

TenurePerAge = Tenure / (Age + 1e-6)
ActiveMember_And_MultiProduct = IsActiveMember * IsMultiProductCustomer


##### Creating a DataFrame for the input data to apply the preprocessor #####
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary],
    'BalanceToSalaryRatio': [BalanceToSalaryRatio],
    'IsHighBalance': [IsHighBalance],
    'IsMultiProductCustomer': [IsMultiProductCustomer],
    'AgeGroup': [AgeGroup],
    'CreditScoreBand': [CreditScoreBand],
    'TenurePerAge': [TenurePerAge],
    'ActiveMember_And_MultiProduct': [ActiveMember_And_MultiProduct]
}, columns=columns_name)


######## Preprocessing the input data #####
processed_input = preprocessor.transform(input_data).astype(np.float32)


###### Making predictions #####
prediction = model.predict(processed_input)
prediction_probability=prediction[0][0]

y_pred = (prediction_probability > 0.28).astype(int)

if y_pred== 1:
    st.write("The customer is likely to churn.[Probability of churn: {:.2f}%]".format(prediction_probability * 100))
else:
    st.write("The customer is not likely to churn.[Probability of churn: {:.2f}%]".format(prediction_probability * 100))


