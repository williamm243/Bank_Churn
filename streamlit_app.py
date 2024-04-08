import streamlit as st
import pandas as pd
import pickle

# Load the pickled AdaBoost model
with open('ada_model.pkl', 'rb') as f:
    ada_model = pickle.load(f)

# Define a function to make predictions
def predict_exit(features):
    prediction = ada_model.predict(features)
    return prediction

# Streamlit UI
def main():
    st.title('Bank Customer Exit Prediction')

    # Input features
    credit_score = st.slider('Credit Score (300-850)', 300, 850, step=1)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 100, step=1)
    tenure = st.slider('Tenure at the bank (in years)', 0, 10, step=1)
    balance = st.number_input('Balance in account', min_value=0.0)
    num_of_products = st.slider('Number of Products/accounts', 1, 4, step=1)
    has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
    is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

    # Convert categorical variables to numerical labels
    geography_label = 1 if geography == 'Germany' else (2 if geography == 'Spain' else 0)
    gender_label = 1 if gender == 'Male' else 0
    has_cr_card_label = 1 if has_cr_card == 'Yes' else 0
    is_active_member_label = 1 if is_active_member == 'Yes' else 0

    # Convert features to DataFrame
    features = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography_label],
        'Gender': [gender_label],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_label],
        'IsActiveMember': [is_active_member_label],
        'EstimatedSalary': [estimated_salary]
    })

    # Make predictions
    if st.button('Predict'):
        prediction = predict_exit(features)
        if prediction[0] == 1:
            st.write('Prediction: Customer will most likely exit.')
        else:
            st.write('Prediction: Customer will most likely stay.')

if __name__ == '__main__':
    main()

