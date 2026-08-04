import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# Import RAG Search Module
from rag_engine import query_rag_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Customer Risk & Regulatory Analytics Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<p class='main-header'>🏦 Enterprise Analytics & Compliance Intelligence Engine</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Integrated Business Analytics (DA/BA), Predictive ML Churn Risk Scoring (DS), and CFPB Regulatory RAG Assistant (AI)</p>", unsafe_allow_html=True)

# --- Load Datasets & Model Artifacts ---
@st.cache_data
def load_data():
    # Detect file flexibly
    data_path = "data/customer_analytics.csv"
    if not os.path.exists(data_path):
        if os.path.exists("data/customer_analytics"):
            data_path = "data/customer_analytics"
        elif os.path.exists("data/customer_analytics.csv.csv"):
            data_path = "data/customer_analytics.csv.csv"
            
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

@st.cache_resource
def load_ml_artifacts():
    with open("models/risk_xgb_model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["model"], artifacts["explainer"], artifacts["features"]

df = load_data()
model, explainer, feature_names = load_ml_artifacts()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Portfolio Filters")
contract_filter = st.sidebar.multiselect(
    "Contract Type:",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

min_tenure, max_tenure = int(df["tenure"].min()), int(df["tenure"].max())
tenure_range = st.sidebar.slider("Tenure (Months):", min_tenure, max_tenure, (min_tenure, max_tenure))

# Filtered Dataframe
filtered_df = df[
    (df["Contract"].isin(contract_filter)) & 
    (df["tenure"].between(tenure_range[0], tenure_range[1]))
]

# --- Main Interface Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📊 Executive BI Analytics (BA/DA)", 
    "🤖 Predictive ML Scoring & SHAP (DS)", 
    "📄 CFPB Compliance RAG Assistant (AI)"
])

# ==============================================================================
# TAB 1: EXECUTIVE BI ANALYTICS
# ==============================================================================
with tab1:
    st.subheader("📈 Real Customer Portfolio Metrics & Churn Insights")
    
    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active Accounts", f"{len(filtered_df):,}")
    churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
    col2.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg Monthly Charges", f"${filtered_df['MonthlyCharges'].mean():,.2f}")
    col4.metric("Avg Tenure (Months)", f"{filtered_df['tenure'].mean():.1f}")
    
    st.divider()
    
    # Visualizations
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### 💳 Monthly Spend Distribution by Contract Type")
        fig_spend = px.box(
            filtered_df, 
            x="Contract", 
            y="MonthlyCharges", 
            color="Churn",
            color_discrete_map={'No': "#10B981", 'Yes': "#EF4444"},
            template="plotly_white"
        )
        st.plotly_chart(fig_spend, use_container_width=True)
        
    with chart_col2:
        st.markdown("##### ⏳ Churn Volume vs Tenure (Months)")
        fig_tenure = px.histogram(
            filtered_df, 
            x="tenure", 
            color="Churn",
            nbins=30,
            color_discrete_map={'No': "#3B82F6", 'Yes': "#DC2626"},
            template="plotly_white"
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    with st.expander("📋 View Ingested Portfolio Dataset"):
        st.dataframe(filtered_df, use_container_width=True)

# ==============================================================================
# TAB 2: PREDICTIVE ML RISK SCORING & SHAP
# ==============================================================================
with tab2:
    st.subheader("🔮 Customer Risk Assessment & Real-Time Inference")
    
    selected_cust_id = st.selectbox("Select Customer ID:", filtered_df["customerID"].values)
    cust_data = df[df["customerID"] == selected_cust_id].iloc[0]
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.info(f"**Tenure:** {cust_data['tenure']} months")
        st.info(f"**Monthly Charges:** ${cust_data['MonthlyCharges']:,.2f}")
    with col_input2:
        st.info(f"**Total Charges:** ${float(cust_data['TotalCharges']):,.2f}")
        st.info(f"**Senior Citizen:** {'Yes' if cust_data['SeniorCitizen'] == 1 else 'No'}")
    with col_input3:
        st.info(f"**Contract:** {cust_data['Contract']}")
        st.info(f"**Internet Service:** {cust_data['InternetService']}")
        
    # Build Input Dict matching trained features
    input_dict = {
        'tenure': cust_data['tenure'],
        'MonthlyCharges': cust_data['MonthlyCharges'],
        'TotalCharges': float(cust_data['TotalCharges']),
        'SeniorCitizen': cust_data['SeniorCitizen'],
        'Contract_One year': 1 if cust_data['Contract'] == 'One year' else 0,
        'Contract_Two year': 1 if cust_data['Contract'] == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if cust_data['InternetService'] == 'Fiber optic' else 0,
        'InternetService_No': 1 if cust_data['InternetService'] == 'No' else 0,
    }
    
    input_df = pd.DataFrame([input_dict])[feature_names]
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    m_col1, m_col2 = st.columns([1, 2])
    
    with m_col1:
        st.markdown("#### 🎯 Model Churn Risk Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#DC2626" if prob > 0.5 else "#10B981"},
                'steps': [
                    {'range': [0, 50], 'color': "#D1FAE5"},
                    {'range': [50, 100], 'color': "#FEE2E2"}
                ],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with m_col2:
        st.markdown("#### 🧠 SHAP Feature Attribution Plot")
        shap_values = explainer(input_df)
        fig_shap, ax = plt.subplots(figsize=(8, 3))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)

# ==============================================================================
# TAB 3: CFPB COMPLIANCE RAG ASSISTANT
# ==============================================================================
with tab3:
    st.subheader("📄 Federal CFPB TILA & Regulation Z Audit Portal")
    st.markdown("Query the 300+ page **CFPB Truth in Lending Act (TILA / Regulation Z) Examination Manual**:")
    
    user_query = st.text_input(
        "Enter your regulatory query:",
        placeholder="e.g., What is the APR threshold for a loan to be classified as a higher-priced mortgage loan under Regulation Z?"
    )
    
    if user_query:
        with st.spinner("🔍 Querying ChromaDB Vector Engine..."):
            results, engine_used = query_rag_engine(user_query, top_k=2)
            
        st.success(f"⚡ Search Completed via: **{engine_used}**")
        
        for idx, (doc, distance) in enumerate(results, 1):
            # Auto-expand the 1st match so the text is immediately visible
            is_expanded = True if idx == 1 else False
            
            with st.expander(f"📌 Citation Match {idx} | Relevance Distance: {distance:.4f}", expanded=is_expanded):
                st.markdown("**Retrieved Regulatory Passage:**")
                st.info(f"\"{doc.page_content.strip()}\"")
                st.caption(f"📁 Source: {doc.metadata.get('source', 'docs/compliance_policy_2026.pdf')}")