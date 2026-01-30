import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

with open('onehotencoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    l = pickle.load(f)



st.title('Customer Churn Prediction')

geography=st.selectbox('Geography', ohe.categories_[0])
gender=st.selectbox('Gender', l.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.slider('Credit Score')
tenure=st.slider('Tenure', 0, 10)
num_of_products=st.slider('Number of Products', 1, 4)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])
estimated_salary=st.number_input('Estimated Salary')

input_data=pd.DataFrame({
  'CreditScore': [credit_score],
  'Gender': [gender],
  'Age': [age],
  'Tenure': [tenure],
  'Balance': [balance], 
  'NumOfProducts': [num_of_products],
  'HasCrCard': [has_cr_card], 
  'IsActiveMember': [is_active_member],
  'EstimatedSalary': [estimated_salary]
})


geo_ohe=ohe.transform([[geography]]).toarray()
geo_df=pd.DataFrame(geo_ohe,columns=ohe.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)
input_data['Gender'] = l.transform(input_data['Gender'])

input_data_scaled=sc.transform(input_data)
prediction=model.predict(input_data_scaled)
pred=prediction[0][0]

st.write('churn probability:', pred)
if pred>0.5:
  st.write('The customer is likely to churn.')
else:
  st.write('The customer is unlikely to churn.')
