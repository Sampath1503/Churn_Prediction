import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# --- Paths ---
# This code assumes all files are in the *same directory* as app.py
try:
    model_path = "logistic_regression_model.pkl"
    encoder_path = "one_hot_encoder.pkl"
    scaler_path = "scaler.pkl"

    # --- Load files ---
    loaded_model = joblib.load(model_path)
    loaded_ohe = joblib.load(encoder_path)
    loaded_scaler = joblib.load(scaler_path)

    st.write("✅ Model, encoder, and scaler loaded successfully!")

except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.error(f"Fatal Error: Could not find model/preprocessor files. Please ensure 'logistic_regression_model.pkl', 'one_hot_encoder.pkl', and 'scaler.pkl' are in the same GitHub repository directory as 'app.py'.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading files: {e}")
    st.stop()


# Set the title and description of the application
st.title('Customer Churn Prediction App')
st.markdown("""
This application predicts whether a customer is likely to churn.
Please input the customer's details below to get a prediction.
""")

# Define the features
categorical_features = ["state", "voice.plan", "intl.plan"]
numeric_features = [
    "area.code", "account.length", "voice.messages", "intl.mins", "intl.calls",
    "intl.charge", "day.mins", "day.calls", "eve.mins", "eve.calls",
    "night.mins", "night.calls", "customer.calls"
]
# All features (for creating the initial DataFrame)
all_features = categorical_features + numeric_features


# Create input fields for the features
st.header("Customer Information")

# Input fields for categorical features
st.subheader("Categorical Features (Used by Model)")
# Use the categories stored inside the loaded one-hot encoder
input_state = st.selectbox("State", loaded_ohe.categories_[0])
input_voice_plan = st.selectbox("Voice Plan", loaded_ohe.categories_[1])
input_intl_plan = st.selectbox("International Plan", loaded_ohe.categories_[2])

# Input fields for numerical features
st.subheader("Numerical Features (NOTE: These appear to be IGNORED by the model)")
st.warning("Based on the model's structure, the numerical inputs below are not used in the prediction.")

# Arrange numerical inputs in columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    input_area_code = st.number_input("Area Code", min_value=0, value=415, step=1, disabled=True)
    input_voice_messages = st.number_input("Voice Messages", min_value=0, value=0, disabled=True)
    input_intl_calls = st.number_input("International Calls", min_value=0, value=0, disabled=True)
    input_day_calls = st.number_input("Day Calls", min_value=0, value=100, disabled=True)
    input_night_calls = st.number_input("Night Calls", min_value=0, value=100, disabled=True)

with col2:
    input_account_length = st.number_input("Account Length", min_value=0, value=100, disabled=True)
    input_intl_mins = st.number_input("International Mins", min_value=0.0, value=10.0, format="%.2f", disabled=True)
    input_day_mins = st.number_input("Day Mins", min_value=0.0, value=180.0, format="%.2f", disabled=True)
    input_eve_calls = st.number_input("Evening Calls", min_value=0, value=100, disabled=True)
    input_customer_calls = st.number_input("Customer Service Calls", min_value=0, value=1, disabled=True)

with col3:
    input_intl_charge = st.number_input("International Charge", min_value=0.0, value=2.70, format="%.2f", disabled=True)
    input_eve_mins = st.number_input("Evening Mins", min_value=0.0, value=200.0, format="%.2f", disabled=True)
    input_night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0, format="%.2f", disabled=True)


# Add a submit button
submit_button = st.button("Predict Churn")

# --- Prediction Logic (Triggered by button click) ---
if submit_button:
    try:
        # 1. Collect *all* user inputs just to create the DataFrame
        user_input_dict = {
            "state": input_state,
            "voice.plan": input_voice_plan,
            "intl.plan": input_intl_plan,
            "area.code": input_area_code,
            "account.length": input_account_length,
            "voice.messages": input_voice_messages,
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
        }

        # 2. Create a pandas DataFrame from the dictionary
        user_df = pd.DataFrame([user_input_dict], columns=all_features)
        
        # 3. Apply the *exact* (and unusual) preprocessing steps from training
        
        # Get *only* the categorical features from the user's DF
        user_df_categorical = user_df[categorical_features]
        st.write("Input Features (Categorical Only):")
        st.dataframe(user_df_categorical)

        # Apply One-Hot Encoding to categorical features (Produces 55 features)
        user_categorical_encoded = loaded_ohe.transform(user_df_categorical)

        # Apply Scaling to the ONE-HOT-ENCODED features (Produces 55 features)
        # This is the final data the model expects
        user_final_processed = loaded_scaler.transform(user_categorical_encoded) 

        # 4. (No concatenation needed)
        st.write("Processed Data Shape (for debugging):", user_final_processed.shape)


        # 5. Make prediction
        prediction = loaded_model.predict(user_final_processed)
        prediction_proba = loaded_model.predict_proba(user_final_processed)

        # 6. Display the prediction result
        st.subheader("Prediction Result")
        
        prob_churn = prediction_proba[0][1]
        
        if prediction[0] == 'yes':
            st.error(f"This customer is likely to **CHURN** (Probability: {prob_churn:.0%})")
        else:
            st.success(f"This customer is likely to **NOT CHURN** (Churn Probability: {prob_churn:.0%})")

        st.markdown("---")
        st.write(f"Confidence (Churn): {prediction_proba[0][1]:.2f}")
        st.write(f"Confidence (Not Churn): {prediction_proba[0][0]:.2f}")
        st.write("Note: This prediction is based *only* on the customer's State, Voice Plan, and International Plan.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


# Add instructions on how to run the application
st.sidebar.header("How to Run")
st.sidebar.markdown("""
To run this Streamlit application:

1. **Save** this code as `app.py`.
2. **Ensure** you have the files `logistic_regression_model.pkl`, `scaler.pkl`, and `one_hot_encoder.pkl` in the **same directory** as `app.py`.
3. **Open your terminal or command prompt.**
4. **Navigate** to the directory where you saved `app.py`.
5. **Run the command:**
   ```bash
   streamlit run app.py
""")
