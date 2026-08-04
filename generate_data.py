import os
import pandas as pd
import numpy as np
from fpdf import FPDF

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

print("⚡ Generating Enterprise Tabular Dataset for BA/DA/DS Analytics...")

# 1. Generate Structured Customer Dataset
np.random.seed(42)
n_samples = 1000

customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
age = np.random.randint(21, 70, size=n_samples)
monthly_spend = np.round(np.random.uniform(100, 15000, size=n_samples), 2)
support_tickets = np.random.poisson(lam=2, size=n_samples)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n_samples, p=[0.5, 0.3, 0.2])
payment_delays = np.random.randint(0, 5, size=n_samples)
credit_score = np.random.randint(580, 850, size=n_samples)

# Logic for Risk / Churn Probability
risk_score = (
    (support_tickets * 0.15) + 
    (payment_delays * 0.25) - 
    ((credit_score - 580) / 1000) + 
    (np.where(contract_type == "Month-to-Month", 0.3, 0.0)) +
    np.random.normal(0, 0.1, size=n_samples)
)
churn_flag = (risk_score > 0.5).astype(int)

df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Age": age,
    "MonthlySpend": monthly_spend,
    "SupportTickets": support_tickets,
    "ContractType": contract_type,
    "PaymentDelays": payment_delays,
    "CreditScore": credit_score,
    "ChurnRiskFlag": churn_flag
})

df.to_csv("data/customer_analytics.csv", index=False)
print("✅ Created: data/customer_analytics.csv")

# 2. Generate Compliance PDF Document using modern fpdf2 syntax
print("⚡ Generating Regulatory Compliance PDF Document...")

pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", style="B", size=14)

# Header
pdf.cell(0, 10, "Enterprise Regulatory & Compliance Manual - 2026", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(5)

pdf.set_font("helvetica", size=10)

content = """SECTION 1: CUSTOMER RISK ASSESSMENT & AUDITING

1.1 High-Risk Classification Thresholds
Accounts exhibiting more than 3 payment delays within a trailing 12-month window or incurring 4 or more severe support tickets must undergo automated secondary risk review. Month-to-Month contract holders carry an escalated default weighting of 1.3x during quarterly portfolio audits.

1.2 Credit Threshold Guidelines
Customers maintaining credit scores under 620 combined with monthly account spend exceeding $5,000 require mandatory manual underwriting review before contract renewals.

SECTION 2: COMPLIANCE & DATA PRIVACY PROTOCOLS

2.1 Data Retention and Anonymization
All PII (Personally Identifiable Information) ingested through batch ETL pipelines must undergo 256-bit hashing before storage in analytics databases. Automated logs must be purged after 90 continuous business days.

2.2 Dispute Escalation Protocols
When an automated Churn Risk Flag triggers a high-severity review, compliance officers have 48 hours to inspect customer communication logs and issue resolution notes. Failure to review high-risk flags incurs a 5% regulatory non-compliance penalty under Federal Audit Standard 804."""

# Write paragraph blocks safely
for paragraph in content.split("\n\n"):
    pdf.multi_cell(0, 6, paragraph)
    pdf.ln(4)

pdf.output("docs/compliance_policy_2026.pdf")
print("✅ Created: docs/compliance_policy_2026.pdf")
print("🎉 All datasets and compliance manuals generated successfully!")