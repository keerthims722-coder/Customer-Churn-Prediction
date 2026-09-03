"""
Customer Churn Prediction - Interactive Streamlit Web Application
A comprehensive, professional, and student-friendly machine learning demo.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 50%, #1e1b4b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
        margin-bottom: 0px;
    }

    /* KPI Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px 22px;
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #60a5fa;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Risk Outcome Cards */
    .risk-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }
    .risk-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
    }
    .risk-title-high {
        color: #f87171;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }
    .risk-title-low {
        color: #34d399;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 6px;
    }
    .risk-prob {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 12px;
    }

    /* Explanation & Concept Boxes */
    .concept-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 14px 0;
        color: #e2e8f0;
    }
    .concept-box h4 {
        color: #818cf8;
        margin-top: 0;
        margin-bottom: 8px;
        font-weight: 700;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-champion {
        background: #fbbf24;
        color: #78350f;
    }
    .badge-blue {
        background: #3b82f6;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. CACHED ASSET & MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    """Load and return the raw dataset with caching."""
    # Look for dataset in current or parent directory
    paths = [
        "customer_churn_data.csv",
        os.path.join(os.path.dirname(__file__), "customer_churn_data.csv"),
        os.path.join("Task---1-main", "customer_churn_data.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load champion model pipeline and precomputed metadata."""
    model_paths = [
        os.path.join("model", "churn_model_pipeline.joblib"),
        os.path.join(os.path.dirname(__file__), "model", "churn_model_pipeline.joblib"),
    ]
    meta_paths = [
        os.path.join("model", "metadata.joblib"),
        os.path.join(os.path.dirname(__file__), "model", "metadata.joblib"),
    ]
    all_models_paths = [
        os.path.join("model", "all_models.joblib"),
        os.path.join(os.path.dirname(__file__), "model", "all_models.joblib"),
    ]

    pipeline = None
    metadata = None
    all_models = None

    for p in model_paths:
        if os.path.exists(p):
            pipeline = joblib.load(p)
            break

    for p in meta_paths:
        if os.path.exists(p):
            metadata = joblib.load(p)
            break

    for p in all_models_paths:
        if os.path.exists(p):
            all_models = joblib.load(p)
            break

    return pipeline, metadata, all_models


# Initial load
df_raw = load_dataset()
champion_pipeline, metadata, all_models = load_model_artifacts()

# Fallback check
if champion_pipeline is None or metadata is None:
    st.warning("⚠️ Pre-trained model artifacts not found. Automatically training models now...")
    try:
        from train_model import train_and_export_models

        metadata = train_and_export_models()
        champion_pipeline, metadata, all_models = load_model_artifacts()
        st.success("✅ Models trained and exported successfully!")
    except Exception as e:
        st.error(f"Error while training models: {e}")

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px 0 18px 0;">
            <div style="font-size: 2.8rem; margin-bottom: 4px;">🔮</div>
            <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #f8fafc;">ChurnPredict AI</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #94a3b8;">Customer Retention & ML Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    nav_selection = st.radio(
        label="Select Application View:",
        options=[
            "🏠 Dashboard / Overview",
            "🎯 Customer Prediction Studio",
            "📊 Model Comparison & Metrics",
            "🔍 Interactive Data Exploration",
            "📚 ML Concepts & Explanation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🏆 Active Champion Model")
    st.markdown(
        """
        <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid #475569; border-radius: 10px; padding: 12px; font-size: 0.85rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Selected:</span>
                <strong style="color: #fbbf24;">Random Forest</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">ROC-AUC:</span>
                <strong style="color: #34d399;">0.8436</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: #94a3b8;">Accuracy:</span>
                <strong style="color: #60a5fa;">80.91%</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8;">Validation:</span>
                <span style="color: #cbd5e1;">5-Fold Stratified CV</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("🎓 Built for Customer Churn Classification Project Demo")

# -----------------------------------------------------------------------------
# 4. VIEW 1: DASHBOARD / OVERVIEW
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Dashboard / Overview":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Customer Churn Prediction System</div>
            <p class="hero-subtitle">
                An intelligent machine learning application that identifies subscribers at risk of leaving.
                By detecting early warning signs in contract terms, payment habits, and service usage,
                businesses can proactively deploy targeted retention strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metrics Row
    if df_raw is not None:
        total_cust = len(df_raw)
        churn_yes = (df_raw["Churn"] == "Yes").sum()
        churn_no = (df_raw["Churn"] == "No").sum()
        churn_rate = (churn_yes / total_cust) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{total_cust:,}</div>
                    <div class="metric-delta" style="color: #60a5fa;">📋 Full Historical Dataset</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Churned Customers</div>
                    <div class="metric-value" style="color: #f87171;">{churn_yes:,}</div>
                    <div class="metric-delta" style="color: #f87171;">⚠️ Lost Subscribers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Retained Customers</div>
                    <div class="metric-value" style="color: #34d399;">{churn_no:,}</div>
                    <div class="metric-delta" style="color: #34d399;">✅ Active & Loyal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Churn Rate</div>
                    <div class="metric-value" style="color: #fbbf24;">{churn_rate:.2f}%</div>
                    <div class="metric-delta" style="color: #fbbf24;">📊 Class Imbalance Ratio (~1:3)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two Column Layout: Churn Ratio & Business Impact Highlights
    c_left, c_right = st.columns([1, 1.2])

    with c_left:
        st.markdown("### 🍩 Churn Ratio Distribution")
        if df_raw is not None:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Retained (No)", "Churned (Yes)"],
                        values=[churn_no, churn_yes],
                        hole=0.55,
                        marker=dict(colors=["#10b981", "#ef4444"]),
                        textinfo="label+percent",
                        hoverinfo="label+value+percent",
                        textfont=dict(size=14, family="Plus Jakarta Sans"),
                    )
                ]
            )
            fig_pie.update_layout(
                showlegend=False,
                height=320,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    dict(
                        text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:12px;color:#94a3b8;'>Churn Rate</span>",
                        x=0.5,
                        y=0.5,
                        font_size=20,
                        showarrow=False,
                        font_family="Plus Jakarta Sans",
                    )
                ],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with c_right:
        st.markdown("### 💡 Why Churn Prediction Matters to Businesses")
        st.markdown(
            """
            <div class="concept-box">
                <h4>🎯 5x Acquisition Cost Rule</h4>
                Acquiring a new customer costs <b>5 to 7 times more</b> than retaining an existing customer.
                Predictive AI helps focus retention dollars where they produce the highest return.
            </div>
            <div class="concept-box" style="border-left-color: #ec4899;">
                <h4>⚡ Early Risk Detection</h4>
                Customers rarely cancel overnight. Warning signals include <b>Month-to-Month contracts</b>,
                <b>Fiber Optic without technical support</b>, and high monthly billing without loyalty tenure.
            </div>
            <div class="concept-box" style="border-left-color: #10b981;">
                <h4>🤝 Actionable Proactive Retention</h4>
                Instead of reacting after a cancellation notice, account managers can trigger tailored incentives,
                discounts on annual contracts, or complimentary technical support bundles.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔍 Dataset Quick Preview")
    if df_raw is not None:
        with st.expander("📂 View Raw Customer Dataset (First 10 records)", expanded=False):
            st.dataframe(df_raw.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# 5. VIEW 2: CUSTOMER PREDICTION STUDIO
# -----------------------------------------------------------------------------
elif nav_selection == "🎯 Customer Prediction Studio":
    st.markdown(
        """
        <div class="hero-container" style="padding: 22px 28px;">
            <div class="hero-title" style="font-size: 1.85rem;">Customer Churn Prediction Studio</div>
            <p class="hero-subtitle">
                Enter customer profile details below or choose a pre-configured sample customer for an instant college demo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample Customers Fast-Fill Section
    st.markdown("#### ⚡ Quick Demo: Load Pre-configured Sample Profile")
    samples = (
        metadata.get("sample_customers", {})
        if metadata
        else {
            "🟢 Loyal Long-Term Customer (Low Risk)": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 72,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer",
                "MonthlyCharges": 45.00,
                "TotalCharges": 3240.00,
                "description": "High tenure (72 mos), Two-year contract, DSL internet, Tech Support, Bank transfer.",
            },
            "🔴 Month-to-Month Fiber Customer (High Risk)": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.50,
                "TotalCharges": 191.00,
                "description": "Brand new customer (2 mos), Month-to-month, Fiber optic, No Tech Support, Electronic check.",
            },
        }
    )

    sample_cols = st.columns(len(samples))
    for idx, (s_name, s_data) in enumerate(samples.items()):
        with sample_cols[idx]:
            if st.button(
                f"👤 {s_name.split('(')[0].strip()}",
                key=f"sample_btn_{idx}",
                help=s_data.get("description", ""),
                use_container_width=True,
            ):
                for k, v in s_data.items():
                    if k != "description":
                        st.session_state[f"input_{k}"] = v
                st.session_state["active_sample_desc"] = s_data.get("description", "")
                st.toast(f"Loaded: {s_name}", icon="✅")

    if "active_sample_desc" in st.session_state and st.session_state["active_sample_desc"]:
        st.info(f"💡 **Loaded Profile Context:** {st.session_state['active_sample_desc']}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Form
    with st.form(key="prediction_form"):
        st.markdown("### 📋 Enter Customer Attributes")

        col_demo, col_contract, col_services = st.columns(3)

        # 1. Demographics
        with col_demo:
            st.markdown("##### 👤 1. Demographics")
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=0 if st.session_state.get("input_gender", "Female") == "Female" else 1,
                key="form_gender",
            )
            senior_val = st.session_state.get("input_SeniorCitizen", 0)
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=["No (0)", "Yes (1)"],
                index=0 if senior_val == 0 else 1,
                key="form_senior",
            )
            partner = st.selectbox(
                "Partner",
                options=["No", "Yes"],
                index=0 if st.session_state.get("input_Partner", "No") == "No" else 1,
                key="form_partner",
            )
            dependents = st.selectbox(
                "Dependents",
                options=["No", "Yes"],
                index=0 if st.session_state.get("input_Dependents", "No") == "No" else 1,
                key="form_dependents",
            )

        # 2. Account & Billing
        with col_contract:
            st.markdown("##### 💳 2. Contract & Billing")
            contract_opts = ["Month-to-month", "One year", "Two year"]
            c_val = st.session_state.get("input_Contract", "Month-to-month")
            contract = st.selectbox(
                "Contract Type",
                options=contract_opts,
                index=contract_opts.index(c_val) if c_val in contract_opts else 0,
                key="form_contract",
            )
            tenure = st.slider(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(st.session_state.get("input_tenure", 12)),
                key="form_tenure",
            )
            paperless = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=0 if st.session_state.get("input_PaperlessBilling", "Yes") == "Yes" else 1,
                key="form_paperless",
            )
            pm_opts = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            pm_val = st.session_state.get("input_PaymentMethod", "Electronic check")
            payment_method = st.selectbox(
                "Payment Method",
                options=pm_opts,
                index=pm_opts.index(pm_val) if pm_val in pm_opts else 0,
                key="form_payment",
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=150.0,
                value=float(st.session_state.get("input_MonthlyCharges", 70.0)),
                step=1.0,
                key="form_monthly",
            )
            # Default auto-estimated TotalCharges = tenure * MonthlyCharges
            auto_total = (
                float(tenure * monthly_charges)
                if tenure > 0
                else float(st.session_state.get("input_TotalCharges", monthly_charges))
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=float(st.session_state.get("input_TotalCharges", auto_total)),
                step=10.0,
                key="form_total",
            )

        # 3. Subscribed Services
        with col_services:
            st.markdown("##### 🌐 3. Subscribed Services")
            phone_service = st.selectbox(
                "Phone Service",
                options=["Yes", "No"],
                index=0 if st.session_state.get("input_PhoneService", "Yes") == "Yes" else 1,
                key="form_phone",
            )
            multilines_opts = ["No", "Yes", "No phone service"]
            ml_val = st.session_state.get("input_MultipleLines", "No")
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=multilines_opts,
                index=multilines_opts.index(ml_val) if ml_val in multilines_opts else 0,
                key="form_multilines",
            )
            inet_opts = ["DSL", "Fiber optic", "No"]
            inet_val = st.session_state.get("input_InternetService", "Fiber optic")
            internet_service = st.selectbox(
                "Internet Service",
                options=inet_opts,
                index=inet_opts.index(inet_val) if inet_val in inet_opts else 1,
                key="form_inet",
            )
            sec_opts = (
                ["No internet service"]
                if internet_service == "No"
                else ["No", "Yes", "No internet service"]
            )
            sec_val = st.session_state.get("input_OnlineSecurity", "No")
            online_security = st.selectbox(
                "Online Security",
                options=sec_opts,
                index=sec_opts.index(sec_val) if sec_val in sec_opts else 0,
                key="form_sec",
            )
            tech_opts = (
                ["No internet service"]
                if internet_service == "No"
                else ["No", "Yes", "No internet service"]
            )
            tech_val = st.session_state.get("input_TechSupport", "No")
            tech_support = st.selectbox(
                "Tech Support",
                options=tech_opts,
                index=tech_opts.index(tech_val) if tech_val in tech_opts else 0,
                key="form_tech",
            )
            stream_opts = (
                ["No internet service"]
                if internet_service == "No"
                else ["No", "Yes", "No internet service"]
            )
            stream_tv_val = st.session_state.get("input_StreamingTV", "No")
            streaming_tv = st.selectbox(
                "Streaming TV",
                options=stream_opts,
                index=stream_opts.index(stream_tv_val) if stream_tv_val in stream_opts else 0,
                key="form_tv",
            )

            # Additional services in expander
            with st.expander("➕ Additional Services (Backup, Device, Movies)"):
                backup_val = st.session_state.get("input_OnlineBackup", "No")
                online_backup = st.selectbox(
                    "Online Backup",
                    options=sec_opts,
                    index=sec_opts.index(backup_val) if backup_val in sec_opts else 0,
                )
                device_val = st.session_state.get("input_DeviceProtection", "No")
                device_protection = st.selectbox(
                    "Device Protection",
                    options=sec_opts,
                    index=sec_opts.index(device_val) if device_val in sec_opts else 0,
                )
                movies_val = st.session_state.get("input_StreamingMovies", "No")
                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    options=stream_opts,
                    index=stream_opts.index(movies_val) if movies_val in stream_opts else 0,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_submitted = st.form_submit_button(
            "🚀 Predict Customer Churn", use_container_width=True, type="primary"
        )

    # PREDICTION EXECUTION
    if predict_submitted:
        # Build exact 1-row DataFrame required by pipeline
        customer_payload = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": 1 if "Yes" in senior_citizen else 0,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone_service,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet_service,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }
            ]
        )

        try:
            if champion_pipeline is None:
                st.error("Champion model pipeline is not loaded.")
            else:
                prob_churn = float(champion_pipeline.predict_proba(customer_payload)[0][1])
                pred_label = int(champion_pipeline.predict(customer_payload)[0])

                is_high_risk = prob_churn >= 0.50

                st.markdown("---")
                st.markdown("### 📊 Prediction Result & Risk Assessment")

                res_left, res_right = st.columns([1.1, 1])

                with res_left:
                    if is_high_risk:
                        st.markdown(
                            f"""
                            <div class="risk-card-high">
                                <div style="font-size: 3rem; margin-bottom: 8px;">⚠️</div>
                                <div class="risk-title-high">HIGH RISK OF CHURN</div>
                                <div class="risk-prob">Churn Probability: <span style="color:#ef4444; font-size:1.6rem;">{prob_churn*100:.1f}%</span></div>
                                <p style="color: #fca5a5; font-size: 0.95rem; line-height: 1.5;">
                                    This customer exhibits behavioral and billing patterns highly characteristic of churners. Immediate retention action is recommended.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="risk-card-low">
                                <div style="font-size: 3rem; margin-bottom: 8px;">✅</div>
                                <div class="risk-title-low">LOW RISK OF CHURN</div>
                                <div class="risk-prob">Churn Probability: <span style="color:#10b981; font-size:1.6rem;">{prob_churn*100:.1f}%</span></div>
                                <p style="color: #6ee7b7; font-size: 0.95rem; line-height: 1.5;">
                                    This customer demonstrates strong loyalty indicators. Probability of remaining with the service is <b>{(1-prob_churn)*100:.1f}%</b>.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with res_right:
                    # Gauge Meter
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob_churn * 100,
                            number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                                "bar": {"color": "#ef4444" if is_high_risk else "#10b981"},
                                "steps": [
                                    {"range": [0, 40], "color": "rgba(16, 185, 129, 0.2)"},
                                    {"range": [40, 65], "color": "rgba(251, 191, 36, 0.2)"},
                                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.2)"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 3},
                                    "thickness": 0.75,
                                    "value": 50,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(
                        height=240,
                        margin=dict(t=30, b=10, l=30, r=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Plus Jakarta Sans"},
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Risk Factor Breakdown & Actionable Retention Strategy
                st.markdown("#### 🔍 Contributing Risk Analysis & Retention Recommendations")
                risk_factors = []
                pos_factors = []

                if contract == "Month-to-month":
                    risk_factors.append(
                        "**Month-to-month Contract**: Lacks long-term commitment (highest single churn correlation)."
                    )
                else:
                    pos_factors.append(
                        f"**{contract} Contract**: High commitment reduces cancellation probability."
                    )

                if tenure <= 6:
                    risk_factors.append(
                        f"**Short Tenure ({tenure} months)**: New subscribers are in the critical trial window."
                    )
                elif tenure >= 24:
                    pos_factors.append(
                        f"**Long Tenure ({tenure} months)**: Established subscriber with proven loyalty."
                    )

                if payment_method == "Electronic check":
                    risk_factors.append(
                        "**Electronic Check**: Associated with friction and high historical churn rate."
                    )

                if internet_service == "Fiber optic" and tech_support == "No":
                    risk_factors.append(
                        "**Fiber Optic without Tech Support**: Premium high-cost connection without troubleshooting assistance."
                    )

                c_fac1, c_fac2 = st.columns(2)
                with c_fac1:
                    st.markdown("##### ⚠️ Risk Drivers Detected")
                    if risk_factors:
                        for rf in risk_factors:
                            st.write(f"- {rf}")
                    else:
                        st.write("✓ No major negative risk triggers detected.")

                with c_fac2:
                    st.markdown("##### 💡 Recommended Next Actions")
                    if is_high_risk:
                        st.write("1. 🎁 **Offer Contract Incentive**: Pitch a 15% discount for upgrading to a 1-year contract.")
                        st.write("2. 🛠️ **Add Free Tech Support**: Provide 3 months of complimentary technical assistance.")
                        st.write("3. 💳 **Switch Payment Method**: Incentivize setting up automated bank transfer.")
                    else:
                        st.write("1. 🌟 **Loyalty Appreciation**: Enroll customer in VIP reward program.")
                        st.write("2. 📦 **Cross-Sell Opportunity**: Safely introduce complementary add-ons (Backup / Streaming).")

        except Exception as e:
            st.error(f"Error executing prediction: {e}")

# -----------------------------------------------------------------------------
# 6. VIEW 3: MODEL COMPARISON & EVALUATION
# -----------------------------------------------------------------------------
elif nav_selection == "📊 Model Comparison & Metrics":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Model Comparison & Performance Benchmarks</div>
            <p class="hero-subtitle">
                Systematic evaluation of <b>Logistic Regression</b>, <b>Random Forest</b>, and <b>Gradient Boosting</b>
                evaluated with 5-fold Stratified Cross-Validation and held-out test data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metadata is not None:
        cv_res = metadata.get("cv_results", {})
        test_res = metadata.get("test_results", {})
        best_name = metadata.get("best_model_name", "Random Forest")

        # 1. Summary Comparison Table
        st.markdown("### 📋 Evaluation Metrics Summary Table (Test Set)")

        summary_rows = []
        for m_name in test_res.keys():
            t_m = test_res[m_name]
            c_m = cv_res.get(m_name, {})
            summary_rows.append(
                {
                    "Model": f"🏆 {m_name} (Selected)" if m_name == best_name else m_name,
                    "CV ROC-AUC": f"{c_m.get('ROC-AUC', 0):.4f}",
                    "Test ROC-AUC": f"{t_m.get('ROC-AUC', 0):.4f}",
                    "Accuracy": f"{t_m.get('Accuracy', 0)*100:.2f}%",
                    "Precision": f"{t_m.get('Precision', 0):.4f}",
                    "Recall": f"{t_m.get('Recall', 0):.4f}",
                    "F1 Score": f"{t_m.get('F1 Score', 0):.4f}",
                }
            )

        df_metrics = pd.DataFrame(summary_rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Interactive Charts: Bar Comparison & ROC Curves
        col_bar, col_roc = st.columns(2)

        with col_bar:
            st.markdown("### 📊 Metrics Comparison Chart")
            metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            models_list = list(test_res.keys())

            fig_bar = go.Figure()
            palette = ["#3b82f6", "#10b981", "#f59e0b"]

            for i, m_name in enumerate(models_list):
                vals = [test_res[m_name][k] for k in metric_keys]
                fig_bar.add_trace(
                    go.Bar(
                        name=m_name,
                        x=metric_keys,
                        y=vals,
                        marker_color=palette[i % len(palette)],
                        text=[f"{v:.3f}" for v in vals],
                        textposition="auto",
                    )
                )

            fig_bar.update_layout(
                barmode="group",
                height=380,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_roc:
            st.markdown("### 📈 ROC Curves Overlay")
            roc_data = metadata.get("roc_curves_data", {})
            fig_roc = go.Figure()

            # Random baseline
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="#94a3b8", width=1.5),
                    name="Random Baseline (AUC = 0.50)",
                )
            )

            colors_roc = {"Random Forest": "#10b981", "Logistic Regression": "#3b82f6", "Gradient Boosting": "#f59e0b"}
            for m_name, r_info in roc_data.items():
                fig_roc.add_trace(
                    go.Scatter(
                        x=r_info["fpr"],
                        y=r_info["tpr"],
                        mode="lines",
                        line=dict(color=colors_roc.get(m_name, "#a855f7"), width=2.5),
                        name=f"{m_name} (AUC = {r_info['auc']:.4f})",
                    )
                )

            fig_roc.update_layout(
                height=380,
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall)",
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Confusion Matrix Breakdown
        st.markdown("### 🔲 Confusion Matrices (Held-out Test Set)")
        cms = metadata.get("confusion_matrices", {})
        cm_cols = st.columns(len(cms))

        for idx, (m_name, cm_vals) in enumerate(cms.items()):
            with cm_cols[idx]:
                cm_arr = np.array(cm_vals)
                tn, fp, fn, tp = cm_arr.ravel()
                fig_cm = px.imshow(
                    cm_arr,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=["No Churn", "Churn"],
                    y=["No Churn", "Churn"],
                    text_auto=True,
                    color_continuous_scale="Blues" if m_name != best_name else "Greens",
                )
                fig_cm.update_layout(
                    title=f"<b>{m_name}</b>",
                    height=280,
                    margin=dict(t=40, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                st.caption(
                    f"TN: **{tn}** | FP: **{fp}** | FN: **{fn}** | TP: **{tp}** (Sensitivity: **{tp/(tp+fn):.1%}**)"
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. Top Feature Importance
        st.markdown("### 🌲 Top 15 Feature Importances (Random Forest)")
        feat_df = metadata.get("feature_importance_df", None)
        if feat_df is not None:
            top15 = feat_df.head(15).sort_values("Importance", ascending=True)
            fig_fi = go.Figure(
                go.Bar(
                    x=top15["Importance"],
                    y=top15["Feature"],
                    orientation="h",
                    marker=dict(
                        color=top15["Importance"],
                        colorscale="Viridis",
                    ),
                    text=[f"{v:.4f}" for v in top15["Importance"]],
                    textposition="auto",
                )
            )
            fig_fi.update_layout(
                height=450,
                xaxis_title="Gini Feature Importance",
                yaxis_title="Feature Name",
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. VIEW 4: INTERACTIVE DATA EXPLORATION (EDA)
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Interactive Data Exploration":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Interactive Exploratory Data Analysis (EDA)</div>
            <p class="hero-subtitle">
                Explore patterns and statistical distributions in customer demographics, contract commitments,
                and service subscriptions that distinguish churners from loyal subscribers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_raw is not None:
        eda_df = df_raw.copy()
        eda_df["TotalCharges"] = pd.to_numeric(eda_df["TotalCharges"], errors="coerce")

        eda_tab1, eda_tab2, eda_tab3 = st.tabs(
            ["📊 Categorical Churn Drivers", "📈 Numerical Distributions", "🔥 Correlation Heatmap"]
        )

        with eda_tab1:
            st.markdown("#### Analyze Churn Rates by Categorical Attributes")
            cat_choice = st.selectbox(
                "Select Categorical Feature to Explore:",
                options=[
                    "Contract",
                    "InternetService",
                    "PaymentMethod",
                    "TechSupport",
                    "OnlineSecurity",
                    "PaperlessBilling",
                    "SeniorCitizen",
                    "gender",
                    "Partner",
                    "Dependents",
                ],
                index=0,
            )

            # Grouped breakdown
            cat_summary = (
                eda_df.groupby([cat_choice, "Churn"]).size().reset_index(name="Customer Count")
            )
            total_per_group = eda_df.groupby(cat_choice).size().to_dict()
            cat_summary["Churn Rate (%)"] = cat_summary.apply(
                lambda r: (r["Customer Count"] / total_per_group[r[cat_choice]]) * 100, axis=1
            )

            c_fig1, c_fig2 = st.columns(2)

            with c_fig1:
                fig_cat_count = px.bar(
                    cat_summary,
                    x=cat_choice,
                    y="Customer Count",
                    color="Churn",
                    barmode="group",
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    title=f"Customer Volume by {cat_choice}",
                )
                fig_cat_count.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    font=dict(family="Plus Jakarta Sans"),
                )
                st.plotly_chart(fig_cat_count, use_container_width=True)

            with c_fig2:
                churn_only = cat_summary[cat_summary["Churn"] == "Yes"].sort_values(
                    "Churn Rate (%)", ascending=False
                )
                fig_rate = px.bar(
                    churn_only,
                    x=cat_choice,
                    y="Churn Rate (%)",
                    color="Churn Rate (%)",
                    color_continuous_scale="Reds",
                    text_auto=".1f",
                    title=f"Churn Rate (%) by {cat_choice}",
                )
                fig_rate.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[0, 100]),
                    font=dict(family="Plus Jakarta Sans"),
                )
                st.plotly_chart(fig_rate, use_container_width=True)

        with eda_tab2:
            st.markdown("#### Numerical Distributions across Churn Groups")
            num_choice = st.radio(
                "Select Numerical Metric:",
                options=["tenure", "MonthlyCharges", "TotalCharges"],
                horizontal=True,
            )

            n_col1, n_col2 = st.columns(2)

            with n_col1:
                fig_hist = px.histogram(
                    eda_df,
                    x=num_choice,
                    color="Churn",
                    marginal="box",
                    barmode="overlay",
                    opacity=0.7,
                    color_discrete_map={"No": "#10b981", "Yes": "#ef4444"},
                    title=f"{num_choice} Histogram & Boxplot by Churn",
                )
                fig_hist.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    font=dict(family="Plus Jakarta Sans"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with n_col2:
                fig_scatter = px.scatter(
                    eda_df,
                    x="tenure",
                    y="MonthlyCharges",
                    color="Churn",
                    color_discrete_map={"No": "rgba(16,185,129,0.5)", "Yes": "rgba(239,68,68,0.8)"},
                    title="Tenure vs Monthly Charges (Colored by Churn)",
                    opacity=0.6,
                )
                fig_scatter.update_layout(
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    font=dict(family="Plus Jakarta Sans"),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        with eda_tab3:
            st.markdown("#### Numerical Features Correlation Heatmap")
            corr_df = eda_df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
            corr_df["Churn"] = (eda_df["Churn"] == "Yes").astype(int)
            corr_matrix = corr_df.corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap with Target (Churn)",
            )
            fig_corr.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. VIEW 5: MODEL & ML EXPLANATIONS (STUDENT GUIDE)
# -----------------------------------------------------------------------------
elif nav_selection == "📚 ML Concepts & Explanation":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Educational ML Guide & Model Concepts</div>
            <p class="hero-subtitle">
                Clear, student-friendly explanations of every step in the Customer Churn Classification pipeline —
                ideal for understanding machine learning concepts and college presentations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("❓ 1. What is Customer Churn and Binary Classification?", expanded=True):
        st.markdown(
            """
            - **Customer Churn:** Refers to when an existing subscriber discontinues their service (cancels their contract).
            - **Binary Classification:** A supervised machine learning problem where the target outcome has exactly two discrete classes:
              - **Class 1 (Positive):** Customer will Churn (`Yes` $\\rightarrow$ `1`)
              - **Class 0 (Negative):** Customer will Remain (`No` $\\rightarrow$ `0`)
            - **Supervised Learning:** The model is trained on labeled historical data containing past customer attributes and whether each customer ended up leaving.
            """
        )

    with st.expander("🛠️ 2. Why is Preprocessing Required?"):
        st.markdown(
            """
            Real-world datasets cannot be directly ingested by raw mathematical machine learning algorithms without preprocessing:
            1. **Dropping Irrelevant Identifiers:** `customerID` was removed because unique arbitrary IDs do not generalize to future customers.
            2. **Type Coercion & Imputation:** `TotalCharges` contained blank spaces for new customers ($0$ tenure). These were coerced to float and missing values were filled using the **median**.
            3. **Feature Separation:** The dataset was split into **Categorical** (textual options like `Contract`, `InternetService`) and **Numerical** (`tenure`, `MonthlyCharges`, `TotalCharges`).
            """
        )

    with st.expander("🔠 3. Why OneHotEncoder and StandardScaler?"):
        st.markdown(
            """
            We encapsulated preprocessing inside Scikit-Learn's `ColumnTransformer`:
            - **OneHotEncoder (Categorical Features):**
              - Converts text categories into binary indicator columns ($0$ or $1$).
              - **Why not LabelEncoding (1, 2, 3)?** Simple integer encoding falsely implies mathematical order (e.g. *Two-year > One-year > Month-to-month*), which biases linear and distance calculations.
            - **StandardScaler (Numerical Features):**
              - Transforms features to have zero mean ($\\mu = 0$) and unit variance ($\\sigma = 1$).
              - **Why?** Features with large values (e.g. `TotalCharges` up to $\\$8,000$) would otherwise dominate features with smaller scales (e.g. `tenure` $0-72$), skewing gradient updates and distance metrics.
            """
        )

    with st.expander("🤖 4. The Three Algorithms Explained"):
        st.markdown(
            """
            | Algorithm | Core Mechanism | Strengths |
            | :--- | :--- | :--- |
            | **Logistic Regression** | Computes a weighted linear combination passed through a Sigmoid (logistic) function: $P(y=1) = \\frac{1}{1 + e^{-z}}$. | Fast, interpretable coefficients, baseline benchmark. |
            | **Random Forest** | An **ensemble of 200 Decision Trees** trained via Bagging (Bootstrap Aggregating) and random feature subsets. | High accuracy, resilient to overfitting, captures complex non-linear feature interactions. |
            | **Gradient Boosting** | An ensemble where trees are built **sequentially**, with each tree designed to correct the residual errors of prior trees. | High predictive power, minimizes loss function iteratively. |
            """
        )

    with st.expander("📐 5. Understanding Evaluation Metrics (Beyond Simple Accuracy)"):
        st.markdown(
            """
            Because the dataset exhibits a **~26.5% churn class imbalance**, simple **Accuracy** alone is misleading:
            - **Accuracy:** $\\frac{TP + TN}{Total}$ — Overall percentage of correct predictions.
            - **Precision:** $\\frac{TP}{TP + FP}$ — Out of all customers predicted to churn, how many *actually* churned? (Avoids wasting retention budget on false alarms).
            - **Recall (Sensitivity):** $\\frac{TP}{TP + FN}$ — Out of all customers who actually left, how many did the model *catch*? (Crucial to prevent losing high-value subscribers unnoticed).
            - **F1 Score:** The harmonic mean of Precision and Recall: $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$.
            - **ROC-AUC (Area Under ROC Curve):** Measures the model's ability to rank churners above non-churners across all possible probability classification thresholds ($0.0$ to $1.0$).
            """
        )

    with st.expander("🏆 6. Why Random Forest was Selected as the Champion Model"):
        st.markdown(
            """
            1. **Superior Discrimination (ROC-AUC ~ 0.8436):** Random Forest achieved the highest ROC-AUC score on the test set, demonstrating the best capability to rank customer churn risk accurately.
            2. **Ensemble Resilience:** Bagging random feature subsets reduces individual tree variance and prevents overfitting on noisy customer records.
            3. **Feature Importance Interpretability:** Provides transparent Gini importance rankings showing exactly which features drive predictions (e.g. `Contract_Month-to-month`, `tenure`, `MonthlyCharges`).
            """
        )
