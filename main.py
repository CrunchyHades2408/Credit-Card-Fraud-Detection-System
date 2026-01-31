# ---- app.py ----
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")

# =======================
# LOAD DATA & MODEL
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

df = load_data()

# (Optional) load your trained model
# model = joblib.load("models/fraud_model.pkl")

# =======================
# SIDEBAR NAVIGATION
# =======================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Data Visualization", "🧠 Model Insights", "🚀 Predict Fraud", "📈 Real-Time Simulation"]
)

# =======================
# PAGE 1: OVERVIEW
# =======================
if page == "🏠 Overview":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("This dashboard analyzes transactions, detects frauds, and visualizes model performance.")
    
    # Key stats
    total_txn = len(df)
    fraud_txn = df["Class"].sum()
    nonfraud_txn = total_txn - fraud_txn
    fraud_percent = (fraud_txn / total_txn) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_txn:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_txn:,}")
    col3.metric("Fraud Percentage", f"{fraud_percent:.4f}%")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# =======================
# PAGE 2: DATA VISUALIZATION
# =======================
elif page == "📊 Data Visualization":
    st.title("📊 Transaction Analytics")
    
    # Histogram of Amount
    st.subheader("Transaction Amount Distribution")
    fig1 = px.histogram(df, x="Amount", nbins=50, title="Distribution of Transaction Amounts", color="Class")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Bar Chart - Fraud vs Non-Fraud
    st.subheader("Fraud vs Non-Fraud Count")
    fraud_count = df["Class"].value_counts().reset_index()
    fraud_count.columns = ["Class", "Count"]  # rename for clarity
    fig2 = px.bar(
    fraud_count,
    x="Class",
    y="Count",
    color="Class",
    text="Count",
    title="Class Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)


    # Boxplot
    st.subheader("Transaction Amount Boxplot by Class")
    fig3 = px.box(df, x="Class", y="Amount", points="all", color="Class", title="Outlier Detection via Boxplot")
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter Plot (PCA-based visualization)
    st.subheader("PCA-Based Transaction Visualization (Sample)")
    sample_df = df.sample(3000, random_state=42)
    fig4 = px.scatter(sample_df, x="V1", y="V2", color=sample_df["Class"].map({0: "Non-Fraud", 1: "Fraud"}))
    st.plotly_chart(fig4, use_container_width=True)

# =======================
# PAGE 3: MODEL INSIGHTS
# =======================
elif page == "🧠 Model Insights":
    st.title("🧠 Model Performance & Insights")

    st.markdown("""
    Here we analyze model metrics such as Confusion Matrix, Precision–Recall Curve, and Feature Importance.

    """)

    # Example: Confusion Matrix Heatmap (Placeholder)
    cm_data = np.array([[996, 4], [12, 45]])
    fig, ax = plt.subplots()
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.title("Confusion Matrix (Example)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    # Example: Feature Importance (Placeholder)
    st.subheader("Top 10 Important Features ")
    feature_importance = {
        'V14': 0.21, 'V17': 0.18, 'V10': 0.15, 'Amount': 0.12,
        'V12': 0.09, 'V4': 0.08, 'V7': 0.07, 'V11': 0.05,
        'V18': 0.03, 'V2': 0.02
    }
    fi_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
    fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# =======================
# PAGE 4: FRAUD PREDICTION
# =======================
elif page == "🚀 Predict Fraud":
    st.title("🚀 Predict New Transaction Fraud")

    st.markdown("Enter transaction details below and click **Predict Fraud** to get the result:")

    # Example inputs
    amount = st.number_input("Transaction Amount", min_value=0.0, step=10.0)
    v1 = st.number_input("V1", -10.0, 10.0)
    v2 = st.number_input("V2", -10.0, 10.0)
    v3 = st.number_input("V3", -10.0, 10.0)
    v4 = st.number_input("V4", -10.0, 10.0)
    v5 = st.number_input("V5", -10.0, 10.0)

    if st.button("🔎 Predict Fraud"):
        # Example model output (replace with actual model.predict)
        pred = np.random.choice([0, 1], p=[0.95, 0.05])  # Mock prediction
        if pred == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

# =======================
# PAGE 5: REAL-TIME SIMULATION
# =======================
elif page == "📈 Real-Time Simulation":
    st.title("📈 Real-Time Fraud Detection Simulation")

    st.markdown("Simulating incoming transactions... (press Stop to end)")

    start_button = st.button("▶ Start Simulation")
    stop_button = st.button("⏹ Stop")

    placeholder = st.empty()

    if start_button:
        for i in range(1, 21):  # simulate 20 transactions
            time.sleep(0.8)
            transaction_id = 10000 + i
            amount = np.random.uniform(1, 5000)
            pred = np.random.choice(["Legit", "Fraud"], p=[0.97, 0.03])
            color = "🟢" if pred == "Legit" else "🔴"
            placeholder.markdown(f"**Transaction #{transaction_id}** — ₹{amount:.2f} — {color} **{pred}**")
# =======================
# REAL-TIME SIMULATION FIX
# =======================
elif page == "📈 Real-Time Simulation":
    st.title("📈 Real-Time Fraud Detection Simulation")
    st.markdown("Simulating incoming transactions...")

    if 'running' not in st.session_state:
        st.session_state.running = False
        st.session_state.counter = 0

    col1, col2 = st.columns(2)
    if col1.button("▶ Start Simulation"):
        st.session_state.running = True
    if col2.button("⏹ Stop Simulation"):
        st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        # simulate 1 transaction per run
        st.session_state.counter += 1
        transaction_id = 10000 + st.session_state.counter
        amount = np.random.uniform(1, 5000)
        pred = np.random.choice(["Legit", "Fraud"], p=[0.97, 0.03])
        color = "🟢" if pred == "Legit" else "🔴"
        placeholder.markdown(f"**Transaction #{transaction_id}** — ₹{amount:.2f} — {color} **{pred}**")
        
        # rerun after a short delay
        time.sleep(0.8)
        st.experimental_rerun()
    else:
        st.info("Press ▶ Start Simulation to begin.")
