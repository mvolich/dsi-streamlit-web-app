# -*- coding: utf-8 -*-
#import libraries 

import streamlit as st 
import pandas as pd
import joblib

#load model pipeline object

model = joblib.load("model.joblib")

# add title and instructions

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

#Now let's see what we have created in streamlit.

#1: Open Anaconda Prompt and install latest version
#pip install click==8.0.4

#2: Ensure we are in the streamlit virtual environment
#(base) C:\Users\mvoli>conda activate dsi-streamlit-web-app

#3: Point to directory where the code is located
#(dsi-streamlit-web-app) C:\Users\mvoli> cd C:\Personal Files\Data Science Infinity\Machine Learning\Model Building\Streamlit

#4 use streamlit command to run the code locally
#(dsi-streamlit-web-app) C:\Personal Files\Data Science Infinity\Machine Learning\Model Building\Streamlit>streamlit run DSI-streamlit-web-app.py

#Now let's keep going

#age unput form

age = st.number_input(
    label= "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

#gender input form

gender = st.radio(
    label= "02. Enter the customer's gender",
    options = ['M', 'F']
    )

#credit score input form

credit_score = st.number_input(
    label= "03. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500
    )

#sumbit inputs to model

if st.button("Submit For Prediction"):
    
    #store data in DF for prediction
    new_data = pd.DataFrame({"age": [age], "gender": [gender], "credit_score": [credit_score]})
    
    #apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    #outout prediction
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
    
    
### The local host address is http://localhost:8501/