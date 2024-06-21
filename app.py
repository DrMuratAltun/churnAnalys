import streamlit as st
import pandas as pd
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Modeli ve dönüştürücüleri yükleyelim
model = tf.keras.models.load_model('churn_model.keras')

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

def preprocess_data(input_data):
    # Kategorik verileri etiketleyelim
    input_data['Geography'] = le.transform(input_data['Geography'])
    input_data['Gender'] = le.transform(input_data['Gender'])
    
    # Özellikleri ölçekleyelim
    input_data = scaler.transform(input_data)
    return input_data

st.title('Churn Prediction App')

with st.form(key='churn_form'):
    customer_id = st.text_input('Customer ID')
    surname = st.text_input('Surname')
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
    geography = st.selectbox('Geography', options=le.classes_)
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
    balance = st.number_input('Balance', min_value=0.0, value=0.0)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox('Has Credit Card', options=[0, 1])
    is_active_member = st.selectbox('Is Active Member', options=[0, 1])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    
    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_data = pd.DataFrame({
        'CustomerId': [customer_id],
        'Surname': [surname],
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    preprocessed_data = preprocess_data(input_data.drop(['CustomerId', 'Surname'], axis=1))
    prediction = model.predict(preprocessed_data)
    prediction = np.round(prediction).astype(int)[0][0]

    st.write(f'Prediction for Customer {customer_id} ({surname}): {"Exited" if prediction == 1 else "Not Exited"}')
