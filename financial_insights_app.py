import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    type_col = st.selectbox("Select the type column (Credit/Debit, required)", options=data.columns)
    amount_col = st.selectbox("Select the amount/value column", options=[col for col in data.columns if col != type_col])

    # --- Parse and sign the amount using Type column ---
    data[type_col] = data[type_col].astype(str).str.lower().str.strip()
    credit_mask = data[type_col].str.contains('credit|cr|in|deposit')
    debit_mask = data[type_col].str.contains('debit|dr|out|withdrawal')
    data['amount_signed'] = np.where(credit_mask, abs(pd.to_numeric(data[amount_col], errors='coerce')),
                                     -abs(pd.to_numeric(data[amount_col], errors='coerce')))
    amount_col = 'amount_signed'

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    data = data.dropna(subset=[amount_col])
    data['month'] = data[date_col].dt.to_period('M')

    # --- Categorization ---
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

    # --- Example: Show summary ---
    st.markdown("## 📈 Summary Statistics")
    st.metric("Total Records", len(data))
    st.metric("Total Net Flow", f"{data[amount_col].sum():,.2f}")
    st.metric("Total Income", f"{data[data[amount_col] > 0][amount_col].sum():,.2f}")
    st.metric("Total Expenses", f"{-data[data[amount_col] < 0][amount_col].sum():,.2f}")

    # --- Example: Pie chart of expenses ---
    expense_data = data[(data[amount_col] < 0) & (~data['category'].isin(['Salary/Income', 'Other Income']))]
    expense_cats = expense_data.groupby('category')[amount_col].sum().abs().sort_values(ascending=False)
    if not expense_cats.empty:
        fig_pie, ax_pie = plt.subplots()
        expense_cats.plot.pie(autopct='%1.1f%%', ax=ax_pie, colormap='tab20')
        plt.ylabel('')
        st.pyplot(fig_pie)
    else:
        st.info("No expenses found for pie chart.")

else:
    st.info("👆 Please upload one or more CSV files to begin.")
         
