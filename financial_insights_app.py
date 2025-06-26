import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet

# --- Custom CSS for a modern look ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .block-container {padding-top: 2rem;}
    .stButton>button {color: white; background: #4F8BF9;}
    .stMetric {background: #e3f2fd; border-radius: 8px;}
    .stTabs [data-baseweb="tab-list"] {justify-content: center;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/color/96/000000/bank.png", width=80)
st.sidebar.title("💡 About This App")
st.sidebar.info(
    "This dashboard provides AI-powered financial insights and predictions from your bank statements. "
    "Upload your CSV files to get started! Works with SBI, ICICI, PNB, APGB, and more."
)
st.sidebar.markdown("[GitHub Repo](https://github.com/Challakruthika/data_bank)")

st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>📊 Financial Insights & AI Predictions</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #888;'>Upload your bank statement CSV files below</h4>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload one or more bank statement CSV files",
    type="csv",
    accept_multiple_files=True
)

data = None
if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        dfs = []
        for file in uploaded_files:
            try:
                df = pd.read_csv(file)
                df['source_file'] = file.name
                dfs.append(df)
            except Exception as e:
                st.warning(f"Error reading {file.name}: {e}")
        if dfs:
            data = pd.concat(dfs, ignore_index=True)
else:
    st.info("👆 Please upload one or more CSV files to begin.")

if data is not None and not data.empty:
    st.success("✅ Data uploaded successfully!")
    with st.expander("🔍 Preview Uploaded Data"):
        st.dataframe(data.head())
        st.write("Columns detected:", list(data.columns))

    st.markdown("### 🛠 Select Columns")
    date_col = st.selectbox("Select the date column", options=data.columns)
    desc_col = st.selectbox("Select the description column (optional)", options=["None"] + list(data.columns))
    type_col = st.selectbox("Select the type column (Credit/Debit/Dr/Cr, optional)", options=["None"] + list(data.columns))

    use_separate = st.checkbox("My statement has separate columns for Deposit and Withdrawal")

    if use_separate:
        deposit_col = st.selectbox("Select the deposit/credit column", options=data.columns)
        withdrawal_col = st.selectbox("Select the withdrawal/debit column", options=data.columns)
        data[deposit_col] = pd.to_numeric(data[deposit_col], errors='coerce').fillna(0)
        data[withdrawal_col] = pd.to_numeric(data[withdrawal_col], errors='coerce').fillna(0)
        data['net_amount'] = data[deposit_col] - data[withdrawal_col]
        amount_col = 'net_amount'
    else:
        amount_col = st.selectbox("Select the amount column (net flow, +ve for credit, -ve for debit)", options=data.columns)
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce').fillna(0)

    # Date parsing and month extraction
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    data['month'] = data[date_col].dt.to_period('M').astype(str)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Trends", "Predictions", "Anomalies"])

    with tab1:
        st.markdown("## 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Total Net Flow", f"{data[amount_col].sum():,.2f}")
        with col3:
            st.metric("Avg. Monthly Net Flow", f"{data.groupby('month')[amount_col].sum().mean():,.2f}")

        if use_separate:
            total_income = data[deposit_col].sum()
            total_expense = data[withdrawal_col].sum()
            savings_rate = 100 * (total_income - total_expense) / total_income if total_income else 0
        else:
            total_income = data[data[amount_col] > 0][amount_col].sum()
            total_expense = -data[data[amount_col] < 0][amount_col].sum()
            savings_rate = 100 * (total_income - total_expense) / total_income if total_income else 0

        st.metric("Total Income", f"{total_income:,.2f}")
        st.metric("Total Expense", f"{total_expense:,.2f}")
        st.metric("Savings Rate (%)", f"{savings_rate:.1f}")

        # Recommendations & Insights (ONLY in tab1)
        st.markdown("## 📝 Recommendations & Insights")
        expense_cats = data.groupby(desc_col)[amount_col].sum().sort_values(ascending=False) if desc_col != "None" else pd.Series()
        if not expense_cats.empty:
            top_cats = expense_cats.head(3)
            st.info(f"Your top spending categories are: {', '.join(top_cats.index)}. Consider reviewing these for savings opportunities.")
        else:
            st.info("Categorized spending insights will appear here if you select a description column.")

    with tab2:
        st.markdown("## 📊 Trends")
        monthly = data.groupby('month')[amount_col].sum()
        st.line_chart(monthly)
        st.markdown("### 📌 Insights")
        st.write(f"Best month: {monthly.idxmax()} ({monthly.max():,.2f})")
        st.write(f"Worst month: {monthly.idxmin()} ({monthly.min():,.2f})")
        if desc_col != "None":
            top_cat = data.groupby(desc_col)[amount_col].sum().idxmax()
            st.write(f"Top spending category: {top_cat}")

    with tab3:
        st.markdown("### 🔮 Net Flow Forecast (Next 6 Months)")
        monthly = data.groupby('month')[amount_col].sum()
        df_prophet = monthly.reset_index().rename(columns={'month': 'ds', amount_col: 'y'})
        df_prophet['ds'] = df_prophet['ds'].astype(str)
        m = Prophet()
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)

        fig4 = m.plot(forecast)
        st.pyplot(fig4)

        st.markdown("### 📘 What Does This Forecast Mean?")
        st.write(
            "The forecast graph above shows your expected financial net flow (income minus expenses) "
            "for the next 6 months, based on your historical data."
        )
        st.write("• **yhat**: Predicted net amount (savings/overspending).")
        st.write("• **yhat_lower / yhat_upper**: Range of possible outcomes based on historical variability.")

        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

        next_month_pred = forecast['yhat'].iloc[-1]
        if next_month_pred < 0:
            st.warning(
                '📉 Your predicted net flow for next month is **negative**.\n\n'
                '👉 This means you might spend more than you earn. Consider reducing discretionary expenses or finding ways to increase income.'
            )
        else:
            st.success(
                '📈 Your predicted net flow for next month is **positive**.\n\n'
                '✅ Keep up your good financial habits and savings momentum!'
            )

        st.markdown("### 📅 Summary of Next 6 Months Forecast")
        for i in range(1, 7):
            row = forecast.iloc[-i]
            label = pd.to_datetime(row['ds']).strftime('%B %Y')
            pred = row['yhat']
            if pred >= 0:
                st.markdown(f"✅ **{label}**: Projected Savings of ₹{pred:,.2f}")
            else:
                st.markdown(f"❌ **{label}**: Projected Overspending of ₹{-pred:,.2f}")

        avg_pred = forecast['yhat'].tail(6).mean()
        score = min(max((avg_pred / total_income) * 100, 0), 100) if total_income != 0 else 0
        st.markdown("### 💡 Your Financial Health Score")
        if score >= 75:
            st.success(f"🏆 Excellent! Your score is {score:.1f}%. You're in great financial shape!")
        elif score >= 50:
            st.info(f"👍 Good! Your score is {score:.1f}%. You're doing well, but review your expenses monthly.")
        elif score >= 25:
            st.warning(f"⚠ Caution! Your score is {score:.1f}%. Consider budgeting or cutting down on non-essential spend.")
        else:
            st.error(f"🚨 Critical! Your score is {score:.1f}%. Immediate action needed to avoid financial stress.")

        st.markdown("### 💡 Suggestions")
        if next_month_pred < 0:
            st.markdown("- Review your top 3 spending categories and set monthly budgets for them.")
            st.markdown("- Track your expenses weekly to avoid surprises.")
            st.markdown("- Consider pausing or cancelling unused subscriptions.")
            st.markdown("- Explore ways to increase your income (side gigs, freelancing, etc.).")
        else:
            st.markdown("- Increase your savings or investments for future goals.")
            st.markdown("- Review your expenses to find small, consistent savings.")
            st.markdown("- Plan for upcoming large expenses in advance.")
            st.markdown("- Keep monitoring your finances monthly.")

    with tab4:
        st.markdown("## 🚨 Anomaly Detection")
        clf = IsolationForest(contamination=0.03, random_state=42)
        data['anomaly'] = clf.fit_predict(data[[amount_col]])
        anomalies = data[data['anomaly'] == -1]
        st.dataframe(anomalies)
        st.markdown("### 📌 Insights")
        st.write(f"Number of anomalies detected: {len(anomalies)}")
        if len(anomalies) > 0:
            st.warning("Some transactions are unusual compared to your typical pattern. Review them for errors or fraud.")

# --- Footer with logo and company name ---
st.markdown(
    """
    <div style='text-align: center; margin-top: 40px;'>
        <img src='chenna_ai_logo.png' width='60' style='vertical-align: middle; margin-right: 10px;'/>
        <span style='font-size: 18px; color: #4F8BF9; vertical-align: middle;'>
            Made by <b>Chenna AI Tech Solutions</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
