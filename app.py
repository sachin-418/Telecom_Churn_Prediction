import streamlit as st
import pandas as pd
import joblib
import os

print("Current Folder:", os.getcwd())
print("Files in Folder:", os.listdir())
# ===========================
# Load Model
# ===========================
model = joblib.load("telecom_churn_pipeline.pkl")

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.title("📞 Telecom Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to churn.")

# ===========================
# Customer Information
# ===========================
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Male", "Female"])

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    if phone == "Yes":
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No"]
        )
    else:
        multiple = "No"

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    if internet != "No":

        security = st.selectbox(
            "Online Security",
            ["Yes", "No"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No"]
        )

        device = st.selectbox(
            "Device Protection",
            ["Yes", "No"]
        )

        tech = st.selectbox(
            "Tech Support",
            ["Yes", "No"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No"]
        )

    else:

        security = "No"
        backup = "No"
        device = "No"
        tech = "No"
        tv = "No"
        movies = "No"

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=0.5
    )

# ===========================
# Predict Button
# ===========================

if st.button("Predict"):

    services = [
        phone,
        multiple,
        security,
        backup,
        device,
        tech,
        tv,
        movies
    ]

    num_services = sum(service == "Yes" for service in services)

    input_df = pd.DataFrame({

        "gender": [1 if gender == "Male" else 0],

        "SeniorCitizen": [senior],

        "Partner": [1 if partner == "Yes" else 0],

        "Dependents": [1 if dependents == "Yes" else 0],

        "tenure": [tenure],

        "PhoneService": [1 if phone == "Yes" else 0],

        "MultipleLines": [1 if multiple == "Yes" else 0],

        "InternetService": [internet],

        "OnlineSecurity": [1 if security == "Yes" else 0],

        "OnlineBackup": [1 if backup == "Yes" else 0],

        "DeviceProtection": [1 if device == "Yes" else 0],

        "TechSupport": [1 if tech == "Yes" else 0],

        "StreamingTV": [1 if tv == "Yes" else 0],

        "StreamingMovies": [1 if movies == "Yes" else 0],

        "Contract": [contract],

        "PaperlessBilling": [1 if paperless == "Yes" else 0],

        "PaymentMethod": [payment],

        "MonthlyCharges": [monthly],

        "TotalCharges": [round(monthly * tenure, 2)],

        "NumServices": [num_services],

        "IsNewCustomer": [1 if tenure < 12 else 0]

    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        if prediction == 1:
            st.error("🔴 Likely to Churn")
        else:
            st.success("🟢 Not Likely to Churn")

    with col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col3:

        if probability < 0.20:
            risk = "🟢 Low"

        elif probability < 0.50:
            risk = "🟡 Medium"

        elif probability < 0.75:
            risk = "🟠 High"

        else:
            risk = "🔴 Critical"

        st.metric(
            "Risk Level",
            risk
        )

    st.metric(
        "💰 Revenue At Risk",
        f"₹{monthly:.2f}/month"
    )

    st.subheader("💡 Recommended Action")

    if probability >= 0.75:

        st.error("""
• Immediate retention call

• Offer 20% discount

• Recommend One-Year Contract
""")

    elif probability >= 0.50:

        st.warning("""
• Send Promotional Offer

• Recommend Annual Plan

• Provide Customer Support
""")

    elif probability >= 0.20:

        st.info("""
• Email Engagement Campaign

• Monitor Customer Activity
""")

    else:

        st.success("""
• Customer is Loyal

• No Immediate Action Required
""")