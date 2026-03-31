import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("data/best_model.pkl")
    scaler        = joblib.load("data/scaler.pkl")
    feature_names = joblib.load("data/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Enter customer details below to predict whether they are at risk of churning. "
    "Built with **Logistic Regression** · AUC-ROC: **0.9903**"
)
st.divider()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this app")
    st.markdown("""
    **Dataset:** IBM Telco Customer Churn  
    **Customers:** 7,043  
    **Best model:** Logistic Regression  

    **Model performance:**
    | Metric | Score |
    |---|---|
    | Accuracy | 94.32% |
    | F1 Score | 0.8969 |
    | AUC-ROC  | 0.9903 |

    **Top churn drivers:**
    1. Satisfaction Score
    2. Number of Referrals
    3. Online Security

    ---
    Built by **Saivarun Kotha**  
    MS Data Science @ UMBC
    """)

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Customer details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Account info**")
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    contract        = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total_charges   = st.number_input("Total charges ($)", min_value=0.0, value=float(tenure * monthly_charges))
    paperless       = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method  = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.markdown("**Services**")
    phone_service   = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet        = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    device_protect  = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies= st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

with col3:
    st.markdown("**Demographics**")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    age             = st.slider("Age", 18, 90, 35)
    senior          = st.selectbox("Senior citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["Yes", "No"])
    dependents      = st.selectbox("Dependents", ["Yes", "No"])
    married         = st.selectbox("Married", ["Yes", "No"])
    satisfaction    = st.slider("Satisfaction score (1=low, 5=high)", 1, 5, 3)
    num_referrals   = st.slider("Number of referrals", 0, 10, 0)
    num_dependents  = st.slider("Number of dependents", 0, 5, 0)

st.divider()

# ── Build input dataframe ─────────────────────────────────────────────────────
def build_input():
    yes_no = lambda v: 1 if v == "Yes" else 0
    svc    = lambda v: 1 if v == "Yes" else 0  # No internet service → 0

    # Tenure group
    if tenure <= 12:
        tenure_group = 0
    elif tenure <= 24:
        tenure_group = 1
    elif tenure <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    # Num services
    services = [online_security, online_backup, device_protect,
                tech_support, streaming_tv, streaming_movies]
    num_services = sum(1 for s in services if s == "Yes")

    # Revenue per GB (approximate — no GB data in form, use 10 as default)
    revenue_per_gb = monthly_charges / 10.0

    # Contract encoding
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    payment_map  = {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    }
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    gender_map   = {"Female": 0, "Male": 1}
    lines_map    = {"No": 0, "No phone service": 1, "Yes": 2}

    row = {
        "Gender":               gender_map[gender],
        "Senior Citizen":       yes_no(senior),
        "Partner":              yes_no(partner),
        "Dependents":           yes_no(dependents),
        "Tenure Months":        tenure,
        "Phone Service":        yes_no(phone_service),
        "Multiple Lines":       lines_map[multiple_lines],
        "Internet Service":     internet_map[internet],
        "Online Security":      svc(online_security),
        "Online Backup":        svc(online_backup),
        "Device Protection":    svc(device_protect),
        "Tech Support":         svc(tech_support),
        "Streaming TV":         svc(streaming_tv),
        "Streaming Movies":     svc(streaming_movies),
        "Contract":             contract_map[contract],
        "Paperless Billing":    yes_no(paperless),
        "Payment Method":       payment_map[payment_method],
        "Monthly Charges":      monthly_charges,
        "Total Charges":        total_charges,
        "Age":                  age,
        "Married":              yes_no(married),
        "Number of Dependents": num_dependents,
        "Satisfaction Score":   satisfaction,
        "Number of Referrals":  num_referrals,
        "Tenure Group":         tenure_group,
        "Num Services":         num_services,
        "Revenue per GB":       revenue_per_gb,
    }

    # Build dataframe aligned to training features
    input_df = pd.DataFrame([row])

    # Add any missing columns with 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only training columns in correct order
    input_df = input_df[feature_names]
    return input_df

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("Predict churn risk", type="primary", use_container_width=True):

    input_df = build_input()

    # Predict
    prob      = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.5)

    st.divider()
    st.subheader("Prediction result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ HIGH CHURN RISK")
        else:
            st.success("✅ LOW CHURN RISK")

    with res_col2:
        st.metric("Churn probability", f"{prob:.1%}")

    with res_col3:
        st.metric("Retention probability", f"{1 - prob:.1%}")

    # Risk gauge
    st.markdown("**Risk level**")
    color = "🔴" if prob > 0.7 else "🟠" if prob > 0.4 else "🟢"
    st.progress(float(prob))
    st.caption(f"{color} Churn probability: {prob:.1%}")

    # Actionable recommendation
    st.divider()
    st.subheader("Recommended action")

    if prob > 0.7:
        st.error("""
        **Immediate action required.**  
        This customer is at very high risk. Consider:
        - Offer a contract upgrade with a discount
        - Assign a dedicated support representative
        - Provide a free service upgrade (e.g. online security)
        """)
    elif prob > 0.4:
        st.warning("""
        **Monitor closely.**  
        This customer shows moderate churn signals. Consider:
        - Send a satisfaction survey
        - Offer a loyalty reward or referral incentive
        - Check if they have online security enabled
        """)
    else:
        st.success("""
        **Customer appears loyal.**  
        Low churn risk. Consider:
        - Encourage referrals — they reduce churn further
        - Offer a contract upgrade to lock in retention
        """)

    # Show model charts
    st.divider()
    st.subheader("Model insights")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image("data/shap_feature_importance.png",
                 caption="Top features driving churn (SHAP)", use_column_width=True)
    with img_col2:
        st.image("data/confusion_matrix.png",
                 caption="Model confusion matrix", use_column_width=True)

    st.image("data/roc_curves.png",
             caption="ROC curves — all 3 models", use_column_width=True)
