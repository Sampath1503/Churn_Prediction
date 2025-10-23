import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Define the directory where the model and preprocessing objects are saved
model_dir = 'churn_prediction_model'

# Load the saved model, scaler, and one-hot encoder
loaded_model = joblib.load(os.path.join(model_dir, 'logistic_regression_model.pkl'))
loaded_scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
loaded_ohe = joblib.load(os.path.join(model_dir, 'one_hot_encoder.pkl'))


# Set the title and description of the application
st.title('Customer Churn Prediction App')
st.markdown("""
This application predicts whether a customer is likely to churn based on their telecommunications usage patterns.
Please input the customer's details below to get a prediction.
""")

# Define the features the model expects (excluding 'Unnamed: 0' and dropped correlated features)
# Based on the notebook, the features used for encoding and training were:
# 'state', 'area.code', 'account.length', 'voice.plan', 'voice.messages',
# 'intl.plan', 'intl.mins', 'intl.calls', 'intl.charge', 'day.mins',
# 'day.calls', 'eve.mins', 'eve.calls', 'night.mins', 'night.calls',
# 'customer.calls'
# From these, 'state', 'voice.plan', and 'intl.plan' were one-hot encoded.
# The numeric features were the remaining ones.

categorical_features = ["state", "voice.plan", "intl.plan"]
numeric_features = [
    "area.code", "account.length", "voice.messages", "intl.mins", "intl.calls",
    "intl.charge", "day.mins", "day.calls", "eve.mins", "eve.calls",
    "night.mins", "night.calls", "customer.calls"
]


# Create input fields for the features
st.header("Customer Information")

# Input fields for categorical features
st.subheader("Categorical Features")
input_state = st.selectbox("State", loaded_ohe.categories_[0])
input_voice_plan = st.selectbox("Voice Plan", loaded_ohe.categories_[1])
input_intl_plan = st.selectbox("International Plan", loaded_ohe.categories_[2])

# Input fields for numerical features
st.subheader("Numerical Features")

# Arrange numerical inputs in columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    input_area_code = st.number_input("Area Code", min_value=0, value=415, step=1)
    input_voice_messages = st.number_input("Voice Messages", min_value=0, value=0)
    input_intl_calls = st.number_input("International Calls", min_value=0, value=0)
    input_day_calls = st.number_input("Day Calls", min_value=0, value=100)
    input_night_calls = st.number_input("Night Calls", min_value=0, value=100)


with col2:
    input_account_length = st.number_input("Account Length", min_value=0, value=100)
    input_intl_mins = st.number_input("International Mins", min_value=0.0, value=10.0)
    input_day_mins = st.number_input("Day Mins", min_value=0.0, value=180.0)
    input_eve_calls = st.number_input("Evening Calls", min_value=0, value=100)
    input_customer_calls = st.number_input("Customer Service Calls", min_value=0, value=1)


with col3:
    input_intl_charge = st.number_input("International Charge", min_value=0.0, value=2.70)
    input_eve_mins = st.number_input("Evening Mins", min_value=0.0, value=200.0)
    input_night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0)


# Add a submit button
submit_button = st.button("Predict Churn")

# --- Prediction Logic (Triggered by button click) ---
if submit_button:
    # 1. Collect user inputs into a dictionary
    user_input_dict = {
        "state": input_state,
        "area.code": input_area_code,
        "account.length": input_account_length,
        "voice.plan": input_voice_plan,
        "voice.messages": input_voice_messages,
        "intl.plan": input_intl_plan,
        "intl.mins": input_intl_mins,
        "intl.calls": input_intl_calls,
        "intl.charge": input_intl_charge,
        "day.mins": input_day_mins,
        "day.calls": input_day_calls,
        "eve.mins": input_eve_mins,
        "eve.calls": input_eve_calls,
        "night.mins": input_night_mins,
        "night.calls": input_night_calls,
        "customer.calls": input_customer_calls,
        # Note: 'Unnamed: 0', 'day.charge', 'eve.charge', 'night.charge' were dropped
    }

    # 2. Create a pandas DataFrame from the dictionary
    # Define the column order explicitly based on how X_train was structured before encoding
    # This order should match the original features used after dropping 'Unnamed: 0' and the correlated columns.
    # Looking back at the notebook, the columns in X_train (before one-hot encoding) were:
    # ['Unnamed: 0', 'state', 'area.code', 'account.length', 'voice.plan', 'voice.messages',
    #  'intl.plan', 'intl.mins', 'intl.calls', 'intl.charge', 'day.mins', 'day.calls',
    #  'eve.mins', 'eve.calls', 'night.mins', 'night.calls', 'customer.calls']
    # After dropping 'Unnamed: 0', 'day.charge', 'eve.charge', 'night.charge', the order was preserved for the remaining.
    # Let's reconstruct that expected order for the input DataFrame.

    expected_columns_order = [
        'state', 'area.code', 'account.length', 'voice.plan', 'voice.messages',
        'intl.plan', 'intl.mins', 'intl.calls', 'intl.charge', 'day.mins', 'day.calls',
        'eve.mins', 'eve.calls', 'night.mins', 'night.calls', 'customer.calls'
    ]


    user_df = pd.DataFrame([user_input_dict], columns=expected_columns_order)

    st.write("User Input DataFrame (Before Preprocessing):")
    st.dataframe(user_df)

    # 3. Apply the same preprocessing steps

    # Separate categorical and numerical features
    user_df_categorical = user_df[categorical_features]
    user_df_numeric = user_df[numeric_features]

    # Apply One-Hot Encoding to categorical features
    user_df_categorical_encoded = loaded_ohe.transform(user_df_categorical)
    # Convert the encoded array back to a DataFrame for easier handling and column naming
    user_df_categorical_encoded_df = pd.DataFrame(
        user_df_categorical_encoded,
        columns=loaded_ohe.get_feature_names_out(categorical_features)
    )

    # Apply Scaling to numerical features
    # Note: Based on the notebook, scaling was applied to the one-hot encoded features,
    # which is unusual. However, to match the training pipeline exactly, we will apply
    # the scaler to the one-hot encoded features. This might need adjustment based on
    # the actual training pipeline used for the final saved model.
    # Assuming the model was trained on the scaled, one-hot encoded features:
    user_df_final_processed = loaded_scaler.transform(user_df_categorical_encoded_df)

    st.write("Processed User Input (After One-Hot Encoding and Scaling):")
    st.dataframe(pd.DataFrame(user_df_final_processed)) # Displaying as DataFrame for readability

    # 4. Make prediction
    prediction = loaded_model.predict(user_df_final_processed)
    prediction_proba = loaded_model.predict_proba(user_df_final_processed)

    # 5. Display the prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 'yes':
        st.error("This customer is likely to **CHURN**.")
        st.write(f"Probability of Churn: **{prediction_proba[0][1]:.2f}**")
        st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")
    else:
        st.success("This customer is likely to **NOT CHURN**.")
        st.write(f"Probability of Not Churning: **{prediction_proba[0][0]:.2f}**")
        st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")

    st.markdown("---")
    st.write("Note: This prediction is based on the trained Logistic Regression model.")

# Add instructions on how to run the application
st.sidebar.header("How to Run")
st.sidebar.markdown("""
To run this Streamlit application:

1. **Save** the code above as `app.py`.
2. **Ensure** you have Streamlit and necessary libraries (`pandas`, `joblib`, `numpy`, `scikit-learn`, `imblearn`) installed (`pip install streamlit pandas joblib numpy scikit-learn imblearn`).
3. **Ensure** you have the `churn_prediction_model` directory in the same location as `app.py`, containing the saved `logistic_regression_model.pkl`, `scaler.pkl`, and `one_hot_encoder.pkl` files.
4. **Open your terminal or command prompt.**
5. **Navigate** to the directory where you saved `app.py`.
6. **Run the command:**
   ```bash
   streamlit run app.py
   ```

This will open the application in your web browser.
""")
