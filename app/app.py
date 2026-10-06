import sys
import os

import streamlit as st


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.predict import predict_customer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Bank Marketing Campaign Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict whether a customer is likely to subscribe to a term deposit.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

with col2:
    job = st.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown"
        ]
    )

with col3:
    marital = st.selectbox(
        "Marital Status",
        [
            "married",
            "single",
            "divorced"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Education",
        [
            "primary",
            "secondary",
            "tertiary",
            "unknown"
        ]
    )

with col2:
    default = st.selectbox(
        "Credit in Default?",
        [
            "no",
            "yes"
        ]
    )


# --------------------------------------------------
# FINANCIAL INFORMATION
# --------------------------------------------------

st.header("Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    balance = st.number_input(
        "Account Balance",
        value=1500,
        step=100
    )

with col2:
    housing = st.selectbox(
        "Housing Loan?",
        [
            "yes",
            "no"
        ]
    )

with col3:
    loan = st.selectbox(
        "Personal Loan?",
        [
            "no",
            "yes"
        ]
    )


# --------------------------------------------------
# CAMPAIGN INFORMATION
# --------------------------------------------------

st.header("Campaign Information")

col1, col2, col3 = st.columns(3)

with col1:
    contact = st.selectbox(
        "Contact Type",
        [
            "cellular",
            "telephone",
            "unknown"
        ]
    )

with col2:
    day = st.number_input(
        "Last Contact Day",
        min_value=1,
        max_value=31,
        value=15
    )

with col3:
    month = st.selectbox(
        "Last Contact Month",
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec"
        ]
    )


col1, col2, col3 = st.columns(3)

with col1:
    duration = st.number_input(
        "Call Duration (seconds)",
        min_value=0,
        value=300,
        step=10
    )

with col2:
    campaign = st.number_input(
        "Number of Contacts in Campaign",
        min_value=1,
        value=1,
        step=1
    )

with col3:
    previous = st.number_input(
        "Previous Contacts",
        min_value=0,
        value=0,
        step=1
    )


# --------------------------------------------------
# PREVIOUS CAMPAIGN INFORMATION
# --------------------------------------------------

st.header("Previous Campaign Information")

pdays = st.number_input(
    "Days Since Previous Contact",
    min_value=-1,
    value=-1,
    step=1,
    help="-1 means the customer was not previously contacted."
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "other",
        "success"
    ]
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "Predict Subscription",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    customer_data = {
        "age": age,
        "balance": balance,
        "day": day,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome
    }

    try:

        result = predict_customer(customer_data)

        prediction = result["prediction"]
        probability = float(result["probability"])
        threshold = float(result["threshold"])

        st.header("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == "yes":

                st.success(
                    "Customer is likely to subscribe."
                )

            else:

                st.warning(
                    "Customer is unlikely to subscribe."
                )

        with col2:

            st.metric(
                "Subscription Probability",
                f"{probability:.2%}"
            )

        st.progress(
            min(probability, 1.0)
        )

        st.info(
            f"Decision threshold: {threshold:.0%}"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Bank Marketing Campaign Prediction | "
    "Gradient Boosting Machine Learning Model"
)