import joblib
import pandas as pd
import streamlit as st
import numpy as np

# --- 1. Load the *Single* Pipeline File ---
# This file MUST be created from your notebook and include all steps
# (Preprocessor, SMOTE, and the Model)
try:
    pipeline_path = "churn_pipeline.pkl"
    model_pipeline = joblib.load(pipeline_path)
    st.write("✅ Customer Churn Prediction Pipeline loaded successfully!")
except FileNotFoundError:
    st.error(f"Fatal Error: 'churn_pipeline.pkl' not found. Please re-train and upload the pipeline file from your notebook.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the pipeline: {e}")
    st.error("This is often a scikit-learn version mismatch. Check your requirements.txt file.")
    st.stop()


# Set the title and description
st.title('Customer Churn Prediction App')
st.markdown("""
This application predicts whether a customer is likely to churn.
Please input the customer's details below to get a prediction.
""")

# --- 2. Define Features Lists (for input fields) ---
categorical_features = ["state", "voice.plan", "intl.plan"]
numeric_features = [
    "area.code", "account.length", "voice.messages", "intl.mins", "intl.calls",
    "intl.charge", "day.mins", "day.calls", "eve.mins", "eve.calls",
    "night.mins", "night.calls", "customer.calls"
]

# --- 3. Get Dropdown Categories from the Pipeline ---
# This makes your app robust by using the *exact* categories from training
try:
    # Accessing categories from the 'preprocessor' step in the pipeline
    state_categories = model_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[0]
    voice_plan_categories = model_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[1]
    intl_plan_categories = model_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[2]
except Exception as e:
    st.warning(f"Could not load categories dynamically. Using default lists. Error: {e}")
    # Fallback lists if the categories can't be loaded from the pipeline
    state_categories = ['OH', 'NY', 'KS', 'AL', 'WY', 'TX', 'IN'] # Add more if needed
    voice_plan_categories = ['no', 'yes']
    intl_plan_categories = ['no', 'yes']


# --- 4. Create Input Fields ---
st.header("Customer Information")

st.subheader("Categorical Features")
input_state = st.selectbox("State", state_categories)
input_voice_plan = st.selectbox("Voice Plan", voice_plan_categories)
input_intl_plan = st.selectbox("International Plan", intl_plan_categories)

st.subheader("Numerical Features")
col1, col2, col3 = st.columns(3)

with col1:
    input_area_code = st.number_input("Area Code", min_value=0, value=415, step=1)
    input_voice_messages = st.number_input("Voice Messages", min_value=0, value=0)
    input_intl_calls = st.number_input("International Calls", min_value=0, value=0)
    input_day_calls = st.number_input("Day Calls", min_value=0, value=100)
    input_night_calls = st.number_input("Night Calls", min_value=0, value=100)

with col2:
    input_account_length = st.number_input("Account Length", min_value=0, value=100)
    input_intl_mins = st.number_input("International Mins", min_value=0.0, value=10.0, format="%.2f")
    input_day_mins = st.number_input("Day Mins", min_value=0.0, value=180.0, format="%.2f")
    input_eve_calls = st.number_input("Evening Calls", min_value=0, value=100)
    input_customer_calls = st.number_input("Customer Service Calls", min_value=0, value=1)

with col3:
    input_intl_charge = st.number_input("International Charge", min_value=0.0, value=2.70, format="%.2f")
    input_eve_mins = st.number_input("Evening Mins", min_value=0.0, value=200.0, format="%.2f")
    input_night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0, format="%.2f")

# Add a submit button
submit_button = st.button("Predict Churn")

# --- 5. Prediction Logic ---
if submit_button:
    try:
        # 1. Collect user inputs into a dictionary
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
        
        # 2. Create a DataFrame from the dictionary
        # The column order *must match* the order your preprocessor expects
        column_order = numeric_features + categorical_features
        user_df = pd.DataFrame([user_input_dict])[column_order]

        st.write("User Input DataFrame:")
        st.dataframe(user_df)

        # 3. Make prediction
        # The pipeline handles *all* preprocessing (scaling, OHE)
        prediction = model_pipeline.predict(user_df)
        prediction_proba = model_pipeline.predict_proba(user_df)

        # 4. Display result
        st.subheader("Prediction Result")
        
        # Find the index of the 'yes' class
        yes_class_index = np.where(model_pipeline.classes_ == 'yes')[0][0]
        prob_churn = prediction_proba[0][yes_class_index]
        
        if prediction[0] == 'yes':
            st.error(f"This customer is likely to **CHURN** (Probability: {prob_churn:.0%})")
        else:
            st.success(f"This customer is likely to **NOT CHURN** (Churn Probability: {prob_churn:.0%})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Sidebar
st.sidebar.header("How to Run")
st.sidebar.markdown("""
1. **Save** this code as `app.py`.
2. **Ensure** you have the `churn_pipeline.pkl` file (from your notebook) in the **same directory**.
3. **Run:** `streamlit run app.py`
""")
