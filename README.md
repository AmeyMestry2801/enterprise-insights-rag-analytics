Markdown
# 🏦 Enterprise Customer Risk & Regulatory Analytics Platform

A production-grade, multi-modal enterprise analytics platform combining **Predictive Machine Learning (XGBoost + SHAP)**, **Executive BI Analytics (Plotly + Streamlit)**, and an **AI Compliance Engine (ChromaDB + LangChain RAG)**.

This platform empowers Business Analysts, Data Scientists, and Regulatory Compliance Auditors to monitor portfolio churn risk, interpret individual account risk factors, and query official CFPB federal lending regulations in real time.

---

## 🌟 Key Architecture & Capabilities

### 📊 1. Executive BI Analytics (BA / DA)
* **Real-time Portfolio KPI Cards:** Active account volume, historical churn rates, average monthly spend, and customer tenure tracking.
* **Interactive Data Visualizations:** Plotly distribution box plots and tenure histograms segmented by contract type and churn flags.
* **Dynamic Data Filtering:** Sidebar controls to dynamically slice customer cohorts across tenure ranges and contract types.

### 🤖 2. Predictive Risk Engine & SHAP Interpretability (DS / MLE)
* **XGBoost Churn Risk Model:** Trained on real customer behavioral metrics (*tenure, monthly charges, total charges, contract terms, internet tier*), achieving an **ROC-AUC of 0.84+**.
* **Interactive Probability Gauge:** Evaluates individual customer churn probability in real time.
* **Explainable AI (XAI) via SHAP:** Live local waterfall force plots explaining feature attribution and model decision drivers for individual accounts.

### 📄 3. CFPB Regulatory & Compliance RAG Engine (AI / GenAI)
* **Dense Regulatory Semantic Search:** Indexes the official **300+ page CFPB Truth in Lending Act (TILA / Regulation Z)** manual into a local vector store.
* **ChromaDB + HuggingFace Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for dense vector indexing.
* **Automated Fallback Architecture:** Features a local high-speed TF-IDF fallback search engine if neural network embedding pipelines drop.
* **Audit-Ready Citations:** Returns similarity distance metrics and exact regulatory text context for compliance audits.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend / Dashboard:** Streamlit, Plotly, Matplotlib
* **Machine Learning & XAI:** XGBoost, Scikit-Learn, SHAP
* **Vector Store & Embeddings:** ChromaDB, HuggingFace (`all-MiniLM-L6-v2`)
* **Frameworks & Loaders:** LangChain Core, PyPDF, Pandas, NumPy

---

## 📁 Repository Structure

enterprise-insights-rag-analytics/
├── data/
│   └── customer_analytics.csv         # Real Telco Customer Churn Dataset
├── docs/
│   └── compliance_policy_2026.pdf     # CFPB TILA / Regulation Z Manual
├── models/
│   ├── chroma_db/                     # Local ChromaDB Vector Store
│   └── risk_xgb_model.pkl             # Trained XGBoost & SHAP Explainer Artifacts
├── app.py                             # Interactive Streamlit Multi-Tab Application
├── train_model.py                     # XGBoost Model Training & SHAP Serialization Pipeline
├── rag_engine.py                      # Vector Indexing & Semantic Search Engine
├── generate_data.py                   # Data Pipeline Utility
├── requirements.txt                   # Project Dependencies
└── README.md                          # Project Documentation


---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/AmeyMestry2801/enterprise-insights-rag-analytics.git](https://github.com/AmeyMestry2801/enterprise-insights-rag-analytics.git)
cd enterprise-insights-rag-analytics
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Train the XGBoost Model
Bash
python train_model.py
4. Build the ChromaDB Vector Index
Bash
python rag_engine.py
5. Launch the Streamlit Platform
Bash
streamlit run app.py
🔍 Sample RAG Queries to Test
"What is the APR threshold for a loan to be classified as a higher-priced mortgage loan under Regulation Z?"

"What types of transactions are exempt from Regulation Z requirements?"

"What is the required waiting period between providing the Closing Disclosure and loan consummation?"

👤 Author
Amey Mestry

🔗 LinkedIn: linkedin.com/in/ameymestry

🐙 GitHub: github.com/AmeyMestry2801