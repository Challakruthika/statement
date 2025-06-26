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

    use_separate = st.checkbox("My statement has separate columns for Deposit and Withdrawal")

    if use_separate:
        deposit_col = st.selectbox("Select the deposit/credit column", options=data.columns)
        withdrawal_col = st.selectbox("Select the withdrawal/debit column", options=data.columns)
        data[deposit_col] = pd.to_numeric(data[deposit_col], errors='coerce').fillna(0)
        data[withdrawal_col] = pd.to_numeric(data[withdrawal_col], errors='coerce').fillna(0)
        data['net_amount'] = data[deposit_col] - data[withdrawal_col]
        amount_col = 'net_amount'
    else:
        amount_col = st.selectbox("Select the amount column", options=data.columns)
        amount_sign = st.radio(
            "In your selected amount column, what do positive values mean?",
            ("Income/Credit", "Expense/Debit")
        )
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')

        # Auto-detect CR/DR column
        cr_dr_col = None
        for col in data.columns:
            if col.lower().strip() in ["type","txn type","transaction type","cr/dr","dr/cr"]:
                cr_dr_col = col
                break

        if cr_dr_col:
            st.info(f"Detected CR/DR column: **{cr_dr_col}**, auto-adjusting signs.")
            vals = data[cr_dr_col].astype(str).str.upper().str.strip()
            data[amount_col] = np.where(vals.str.startswith("D"),
                                        -data[amount_col].abs(),
                                        data[amount_col].abs())
        else:
            if amount_sign == "Expense/Debit":
                data[amount_col] = -data[amount_col]

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col, amount_col])
    data['month'] = data[date_col].dt.to_period('M')

    def categorize(desc, amt):
        desc = str(desc).lower()
        if amt > 0:
            if any(k in desc for k in ['salary','credit','neft']):
                return 'Salary/Income'
            return 'Other Income'
        else:
            if any(k in desc for k in ['grocery','supermarket','mart']):
                return 'Groceries'
            if any(k in desc for k in ['electric','water','gas','utility']):
                return 'Utilities'
            if 'rent' in desc or 'lease' in desc:
                return 'Rent'
            if 'atm' in desc or 'cash' in desc:
                return 'Cash Withdrawal'
            if any(k in desc for k in ['restaurant','food','cafe']):
                return 'Food & Dining'
            if any(k in desc for k in ['travel','uber','ola','flight']):
                return 'Travel'
            if 'insurance' in desc:
                return 'Insurance'
            if 'emi' in desc or 'loan' in desc:
                return 'Loan/EMI'
            return 'Others'

    if desc_col != "None":
        data['category'] = data.apply(lambda r: categorize(r[desc_col], r[amount_col]), axis=1)
    else:
        data['category'] = data[amount_col].apply(lambda x: 'Other Income' if x > 0 else 'Others')

    data['bank'] = data['source_file'].str.extract(r'(apgb|icici|pnb|sbi)', expand=False).str.upper().fillna('OTHER')

    # === Tabs ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Summary", "📊 Trends", "🔮 Predictions", "🚨 Anomalies", "⬇ Download"
    ])

    with tab1:
        st.markdown("## 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Total Net Flow", f"{data[amount_col].sum():,.2f}")
        with col3:
            st.metric("Avg. Monthly Net Flow", f"{data.groupby('month')[amount_col].sum().mean():,.2f}")
        if use_separate:
            total_income = data[deposit_col].sum()
            total_expense = data[withdrawal_col].sum()
        else:
            total_income = data[data[amount_col] > 0][amount_col].sum()
            total_expense = -data[data[amount_col] < 0][amount_col].sum()
        savings_rate = 100 * (total_income - total_expense) / total_income if total_income else 0
        with col4:
            st.metric("Savings Rate (%)", f"{savings_rate:.2f}")
        st.markdown("### 💵 Income vs. Expense Breakdown")
        st.success(f"Total Income: {total_income:,.2f}")
        st.error(f"Total Expenses: {total_expense:,.2f}")
        expense_data = data[(data[amount_col] < 0) & (~data['category'].isin(['Salary/Income','Other Income']))]
        expense_cats = expense_data.groupby('category')[amount_col].sum().abs().sort_values(ascending=False)
        if not expense_cats.empty:
            fig_pie, ax = plt.subplots()
            expense_cats.plot.pie(autopct='%1.1f%%', ax=ax, colormap='tab20')
            plt.ylabel('')
            st.pyplot(fig_pie)
            top3 = expense_cats.head(3)
            st.info("Top 3 Expense Categories:\n" + "\n".join([f"- {c}: {amt:,.2f}" for c, amt in top3.items()]))
        else:
            st.info("No expenses found.")
        bank_flow = data.groupby('bank')[amount_col].sum().sort_values(ascending=False)
        st.bar_chart(bank_flow)
        if not bank_flow.empty:
            st.info(f"Highest net flow with bank: {bank_flow.idxmax()}")

    with tab2:
        st.markdown("### 📅 Monthly Net Flow")
        monthly = data.groupby('month')[amount_col].sum()
        fig, ax = plt.subplots(figsize=(10,4))
        monthly.plot(kind='bar', color='#4F8BF9', ax=ax)
        plt.ylabel('Net Amount')
        st.pyplot(fig)
        if not monthly.empty:
            diff = monthly.diff().mean()
            st.info(f"Trend is {'increasing' if diff > 0 else 'decreasing'}. Best: {monthly.idxmax()} ({monthly.max():,.2f}), Worst: {monthly.idxmin()} ({monthly.min():,.2f})")
            if (monthly < 0).any():
                st.warning("Some months have negative net flow.")
        cat_month = data.groupby(['month','category'])[amount_col].sum().unstack().fillna(0)
        fig2, ax2 = plt.subplots(figsize=(12,5))
        cat_month.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
        plt.ylabel('Amount')
        st.pyplot(fig2)

    with tab3:
        st.markdown("### 🔮 Forecast Next 6 Months")
        monthly_sum = data.groupby('month')[amount_col].sum().reset_index()
        dfp = monthly_sum.rename(columns={'month':'ds', amount_col:'y'})
        dfp['ds'] = dfp['ds'].astype(str)
        m = Prophet()
        m.fit(dfp)
        future = m.make_future_dataframe(periods=6, freq='M')
        fc = m.predict(future)
        fig3 = m.plot(fc)
        st.pyplot(fig3)
        st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(6))
        nxt = fc['yhat'].iloc[-1]
        st.info(f"Forecast for next month: ₹{nxt:,.2f} ({'good' if nxt>=0 else 'bad'})")

    with tab4:
        st.markdown("### 🚨 Anomalies")
        iso = IsolationForest(contamination=0.01, random_state=42)
        data['anomaly'] = iso.fit_predict(data[[amount_col]])
        anomalies = data[data['anomaly']==-1]
        st.dataframe(anomalies[[date_col,amount_col,'category','source_file']].head(10))
        st.info(f"Detected {len(anomalies)} potential anomalies")

    with tab5:
        st.markdown("### ⬇ Download Cleaned Data & Forecast")
        st.download_button("Cleaned CSV", data.to_csv(index=False), "cleaned_data.csv")
        st.download_button("Forecast CSV", fc[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False), "forecast.csv")

st.markdown("""
<hr style="margin-top:2em; margin-bottom:1em;">
<div style="text-align:center; color: #888;">
Made with ❤ using Streamlit | <a href="https://github.com/Challakruthika/data_bank" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
