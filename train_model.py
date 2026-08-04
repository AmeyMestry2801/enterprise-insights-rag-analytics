import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

os.makedirs("models", exist_ok=True)

print("🚀 Training XGBoost ML Engine on Real Telco Customer Dataset...")

# 1. Flexible File Loading
data_path = "data/customer_analytics.csv"
if not os.path.exists(data_path):
    if os.path.exists("data/customer_analytics"):
        data_path = "data/customer_analytics"
    elif os.path.exists("data/customer_analytics.csv.csv"):
        data_path = "data/customer_analytics.csv.csv"
    else:
        raise FileNotFoundError(f"Could not find dataset in 'data/' directory! Please check your file name.")

print(f"📂 Reading dataset from: '{data_path}'")
df = pd.read_csv(data_path)

# Clean target & total charges column
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Contract', 'PaymentMethod', 'InternetService'], drop_first=True)

# Select numerical + encoded feature columns
features = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
    'Contract_One year', 'Contract_Two year',
    'InternetService_Fiber optic', 'InternetService_No'
]

X = df_encoded[features]
y = df_encoded['Churn']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"📊 Real Data Split: {X_train.shape[0]} Training | {X_test.shape[0]} Test Samples")

# 2. Train Model
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# 3. Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_proba)
print("\n" + "="*50)
print(f"🎯 REAL DATASET ROC-AUC SCORE: {auc_score:.4f}")
print("="*50)
print(classification_report(y_test, y_pred))

# 4. SHAP Explainer
explainer = shap.TreeExplainer(model)

# 5. Serialize Artifacts
artifacts = {
    "model": model,
    "explainer": explainer,
    "features": features
}

with open("models/risk_xgb_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ Model successfully trained and saved!")