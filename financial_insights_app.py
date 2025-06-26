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
        type_col = st.selectbox("Select the type column (Credit/Debit, optional)", options=["None"] + list(data.columns))
        amount_sign = st.radio(
            "In your selected amount column, what do positive values mean?",
            ("Income/Credit", "Expense/Debit")
        )
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')

        if type_col != "None":
            type_vals = data[type_col].astype(str).str.lower().str.strip()
            credit_mask = type_vals.str.contains('credit|cr|in|deposit')
            debit_mask = type_vals.str.contains('debit|dr|out|withdrawal')
            data[amount_col] = np.where(credit_mask, abs(data[amount_col]), -abs(data[amount_col]))
        else:
            if amount_sign == "Expense/Debit":
                data[amount_col] = -data[amount_col]

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    data = data.dropna(subset=[amount_col])
    data['month'] = data[date_col].dt.to_period('M')

    # --- Improved Categorization ---
    def categorize(desc, amt):
        desc = str(desc).lower()
        if amt > 0:
            if 'salary' in desc or 'credit' in desc or 'neft' in desc:
                return 'Salary/Income'
            return 'Other Income'
        else:
            if 'grocery' in desc or 'supermarket' in desc or 'mart' in desc:
                return 'Groceries'
            if 'electric' in desc or 'water' in desc or 'gas' in desc or 'utility' in desc:
                return 'Utilities'
            if 'rent' in desc or 'lease' in desc:
                return 'Rent'
            if 'atm' in desc or 'cash' in desc:
                return 'Cash Withdrawal'
            if 'restaurant' in desc or 'food' in desc or 'cafe' in desc:
                return 'Food & Dining'
            if 'travel' in desc or 'uber' in desc or 'ola' in desc or 'flight' in desc:
                return 'Travel'
            if 'insurance' in desc:
                return 'Insurance'
            if 'emi' in desc or 'loan' in desc:
                return 'Loan/EMI'
            return 'Others'

    if desc_col != "None":
        data['category'] = data.apply(lambda row: categorize(row[desc_col], row[amount_col]), axis=1)
    else:
        data['category'] = data[amount_col].apply(lambda amt: 'Other Income' if amt > 0 else 'Others')

    data['bank'] = data['source_file'].str.extract(r'(apgb|icici|pnb|sbi)', expand=False).str.upper().fillna('OTHER')

    # ... (rest of your code: tabs, charts, insights, etc.) ...

