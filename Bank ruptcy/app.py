import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app
st.title("Bankruptcy Prediction Model")

# Collect user input
import numpy as np
import streamlit as st

# Collect user inputs (you can keep the clean variable names for user interface)
Net_Income_to_Stockholder_Equity = st.number_input('Net Income to Stockholders Equity:', min_value=0.0, max_value=1.0, value=0.84)
Borrowing_dependency = st.number_input('Borrowing Dependency:', min_value=0.0, max_value=100.0, value=0.37)
Persistent_EPS_in_the_Last_Four_Seasons = st.number_input('Persistent EPS in the Last Four Seasons:', min_value=0.0, max_value=1.0, value=0.22)
Net_profit_before_tax_Paid_in_capital = st.number_input('Net profit before tax/Paid-in capital:', min_value=0.0, max_value=1.0, value=0.18)
Liability_to_Equity = st.number_input('Liability to Equity:', min_value=0.0, max_value=1.0, value=0.28)
Degree_of_Financial_Leverage = st.number_input('Degree of Financial Leverage (DFL):', min_value=0.0, max_value=1.0, value=0.02)
Net_Value_Per_Share_A = st.number_input('Net Value Per Share (A):', min_value=0.0, max_value=1.0, value=0.19)
Per_Share_Net_profit_before_tax = st.number_input('Per Share Net profit before tax (Yuan ¥):', min_value=0.0, max_value=1.0, value=0.18)
Net_worth_Assets = st.number_input('Net worth/Assets:', min_value=0.0, max_value=1.0, value=0.89)
Interest_Expense_Ratio = st.number_input('Interest Expense Ratio:', min_value=0.0, max_value=1.0, value=0.63)
Net_Value_Per_Share_B = st.number_input('Net Value Per Share (B):', min_value=0.0, max_value=1.0, value=0.19)
Non_industry_income_and_revenue = st.number_input('Non-industry income and expenditure/revenue:', min_value=0.0, max_value=1.0, value=0.3)
Debt_ratio = st.number_input('Debt ratio %:', min_value=0.0, max_value=1.0, value=0.11)
ROAC_before_interest_and_depreciation_before_interest = st.number_input('ROA(C) before interest and depreciation before interest:', min_value=0.0, max_value=1.0, value=0.50)
Interest_Coverage_Ratio = st.number_input('Interest Coverage Ratio (Interest expense to EBIT):', min_value=0.0, max_value=1.0, value=0.56)

# Make prediction when the user clicks the button
if st.button('Predict'):
    # Create a dictionary to map cleaned variable names to the feature names expected by the model
    input_data_dict = {
        'Net Income to Stockholders Equity': Net_Income_to_Stockholder_Equity,
        'Borrowing Dependency': Borrowing_dependency,
        'Persistent EPS in the Last Four Seasons': Persistent_EPS_in_the_Last_Four_Seasons,
        'Net profit before tax/Paid-in capital': Net_profit_before_tax_Paid_in_capital,
        'Liability to Equity': Liability_to_Equity,
        'Degree of Financial Leverage (DFL)': Degree_of_Financial_Leverage,
        'Net Value Per Share (A)': Net_Value_Per_Share_A,
        'Per Share Net profit before tax (Yuan ¥)': Per_Share_Net_profit_before_tax,
        'Net worth/Assets': Net_worth_Assets,
        'Interest Expense Ratio': Interest_Expense_Ratio,
        'Net Value Per Share (B)': Net_Value_Per_Share_B,
        'Non-industry income and expenditure/revenue': Non_industry_income_and_revenue,
        'Debt ratio %': Debt_ratio,
        'ROA(C) before interest and depreciation before interest': ROAC_before_interest_and_depreciation_before_interest,
        'Interest Coverage Ratio (Interest expense to EBIT)': Interest_Coverage_Ratio
    }

    # Prepare the input vector in the correct order for the model
    input_data = np.array([[
        input_data_dict['Net Income to Stockholders Equity'],
        input_data_dict['Borrowing Dependency'],
        input_data_dict['Persistent EPS in the Last Four Seasons'],
        input_data_dict['Net profit before tax/Paid-in capital'],
        input_data_dict['Liability to Equity'],
        input_data_dict['Degree of Financial Leverage (DFL)'],
        input_data_dict['Net Value Per Share (A)'],
        input_data_dict['Per Share Net profit before tax (Yuan ¥)'],
        input_data_dict['Net worth/Assets'],
        input_data_dict['Interest Expense Ratio'],
        input_data_dict['Net Value Per Share (B)'],
        input_data_dict['Non-industry income and expenditure/revenue'],
        input_data_dict['Debt ratio %'],
        input_data_dict['ROA(C) before interest and depreciation before interest'],
        input_data_dict['Interest Coverage Ratio (Interest expense to EBIT)']
    ]])

    # Predict using the model
    prediction = model.predict(input_data)[0]

    # Display prediction result
    st.write(f'Prediction: {prediction}')
    if prediction==0:
        st.write("Safe to do business")
    else:
        st.write("Bankrupt :  Don't go for business")    


