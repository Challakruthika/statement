import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

st.set_page_config(page_title="Universal Indian Bank Statement Analyzer", layout="wide")
st.title("🇮🇳 Universal Indian Bank Statement Analyzer")
st.markdown("""
Upload your bank statement CSV (SBI, ICICI, PNB, APGB, etc.). The app will auto-detect columns and guide you to map them. Handles Amount, Deposit/Withdrawal, Type (Dr/Cr), and Description-only formats. Get robust insights, trends, forecasts, and actionable tips!
""")

# --- Helper Functions ---
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def detect_date_column(df):
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            return col
        except:
            continue
    return None

def detect_amount_column(df):
    candidates = [c for c in df.columns if 'amount' in c.lower()]
    if candidates:
        return candidates[0]
    # fallback: first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None

def detect_type_column(df):
    for col in df.columns:
        if any(x in col.lower() for x in ['type', 'dr', 'cr', 'debit', 'credit']):
            return col
    return None

def detect_desc_column(df):
    for col in df.columns:
        if 'desc' in col.lower() or 'particular' in col.lower() or 'narration' in col.lower():
            return col
    return None

def detect_deposit_withdrawal_columns(df):
    deposit, withdrawal = None, None
    for col in df.columns:
        if 'deposit' in col.lower() or 'credit' in col.lower():
            deposit = col
        if 'withdrawal' in col.lower() or 'debit' in col.lower():
            withdrawal = col
    return deposit, withdrawal

# --- Keyword-based mapping for Description-only CSVs ---
CREDIT_KEYWORDS = [
    'salary', 'credited', 'deposit', 'neft in', 'upi in', 'imps in', 'refund', 'reversal', 'interest', 'dividend', 'cashback', 'reward', 'pension', 'inward', 'receiv', 'loan disb', 'bonus', 'recd', 'transfer in', 'rtgs in', 'credit', 'cr', 'income', 'paytm add', 'gpay add', 'phonepe add', 'recd', 'recd.']
DEBIT_KEYWORDS = [
    'debited', 'withdrawal', 'atm', 'pos', 'payment', 'emi', 'bill', 'neft out', 'upi out', 'imps out', 'purchase', 'charge', 'fee', 'tax', 'insurance', 'sip', 'ecs', 'mandate', 'transfer out', 'rtgs out', 'debit', 'dr', 'spent', 'paytm pay', 'gpay pay', 'phonepe pay', 'sent', 'sent.']

def classify_desc(desc):
    desc = str(desc).lower()
    if any(k in desc for k in CREDIT_KEYWORDS):
        return 'Credit'
    if any(k in desc for k in DEBIT_KEYWORDS):
        return 'Debit'
    return 'Unknown'

def categorize(desc, amount):
    desc = str(desc).lower()
    if amount > 0:
        # Income categories
        if 'salary' in desc or 'credit' in desc or 'neft' in desc:
            return 'Salary/Income'
        return 'Other Income'
    else:
        # Expense categories
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
        return 'Other Expense'

# --- Main App ---
uploaded_file = st.file_uploader("Upload your bank statement CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    st.write("### Preview", df.head())

    # --- Column Selection ---
    st.write("#### Map your columns")
    date_col = st.selectbox("Date column", options=df.columns, index=df.columns.get_loc(detect_date_column(df)) if detect_date_column(df) else 0)
    desc_col = st.selectbox("Description/Particulars/Narration column", options=[None]+list(df.columns), index=(df.columns.get_loc(detect_desc_column(df))+1) if detect_desc_column(df) else 0)
    type_col = st.selectbox("Type column (Dr/Cr, Debit/Credit, etc.)", options=[None]+list(df.columns), index=(df.columns.get_loc(detect_type_column(df))+1) if detect_type_column(df) else 0)
    amount_col = st.selectbox("Amount column (if single column)", options=[None]+list(df.columns), index=(df.columns.get_loc(detect_amount_column(df))+1) if detect_amount_column(df) else 0)
    deposit_col, withdrawal_col = detect_deposit_withdrawal_columns(df)
    deposit_col = st.selectbox("Deposit/Credit column (if separate)", options=[None]+list(df.columns), index=(df.columns.get_loc(deposit_col)+1) if deposit_col else 0)
    withdrawal_col = st.selectbox("Withdrawal/Debit column (if separate)", options=[None]+list(df.columns), index=(df.columns.get_loc(withdrawal_col)+1) if withdrawal_col else 0)

    # --- Data Preparation ---
    data = pd.DataFrame()
    data['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    if desc_col:
        data['Description'] = df[desc_col].astype(str)
    else:
        data['Description'] = ''

    # --- Logic for Income/Expense/Net Flow ---
    if deposit_col and withdrawal_col:
        data['Credit'] = pd.to_numeric(df[deposit_col], errors='coerce').fillna(0)
        data['Debit'] = pd.to_numeric(df[withdrawal_col], errors='coerce').fillna(0)
    elif amount_col:
        amt = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
        if type_col:
            type_vals = df[type_col].astype(str).str.lower().str.strip()
            # More comprehensive pattern matching for credit/debit types
            credit_mask = type_vals.str.contains('credit|cr|in|credit/|/credit', case=False, na=False)
            debit_mask = type_vals.str.contains('debit|dr|out|debit/|/debit', case=False, na=False)
            
            # Debug info
            st.write(f"Type column values found: {type_vals.unique()}")
            st.write(f"Credit transactions: {credit_mask.sum()}")
            st.write(f"Debit transactions: {debit_mask.sum()}")
            
            data['Credit'] = np.where(credit_mask, amt, 0)
            data['Debit'] = np.where(debit_mask, amt, 0)
        elif desc_col:
            # Keyword-based mapping
            mapped_type = df[desc_col].apply(classify_desc)
            data['Credit'] = np.where(mapped_type == 'Credit', amt, 0)
            data['Debit'] = np.where(mapped_type == 'Debit', amt, 0)
            data['Unknown'] = np.where(mapped_type == 'Unknown', amt, 0)
        else:
            # Fallback: positive = credit, negative = debit
            data['Credit'] = np.where(amt > 0, amt, 0)
            data['Debit'] = np.where(amt < 0, -amt, 0)
    else:
        st.error("Please select at least an Amount column or Deposit/Withdrawal columns.")
        st.stop()

    data = data.sort_values('Date').reset_index(drop=True)
    data = data.dropna(subset=['Date'])

    # --- Metrics ---
    total_income = data['Credit'].sum()
    total_expense = data['Debit'].sum()
    net_flow = total_income - total_expense
    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income", f"₹{total_income:,.2f}")
    col2.metric("Total Expenses", f"₹{total_expense:,.2f}")
    col3.metric("Net Flow", f"₹{net_flow:,.2f}", delta=f"{savings_rate:.1f}% Savings Rate")
    col4.metric("Unknown/Unmapped", f"₹{data['Unknown'].sum():,.2f}" if 'Unknown' in data else "₹0.00")

    # --- Insights ---
    st.write("#### Insights & Recommendations")
    if net_flow < 0:
        st.warning("Your expenses exceed your income. Consider reducing discretionary spending.")
    elif savings_rate < 10:
        st.warning("Your savings rate is below 10%. Try to save at least 20% of your income.")
    else:
        st.success("Good job! Your savings rate is healthy.")
    st.info(f"You have {len(data)} transactions from {data['Date'].min().date()} to {data['Date'].max().date()}.")

    # --- Expense Breakdown ---
    st.write("#### Expense Breakdown by Description (Top 10)")
    if 'Description' in data:
        exp_by_desc = data.groupby('Description')['Debit'].sum().sort_values(ascending=False).head(10)
        fig1 = px.pie(values=exp_by_desc.values, names=exp_by_desc.index, title='Top 10 Expense Categories')
        st.plotly_chart(fig1, use_container_width=True)

    # --- Monthly Trends ---
    st.write("#### Monthly Income & Expense Trends")
    data['Month'] = data['Date'].dt.to_period('M').astype(str)
    monthly = data.groupby('Month').agg({'Credit':'sum', 'Debit':'sum'}).reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly['Month'], y=monthly['Credit'], name='Income', marker_color='green'))
    fig2.add_trace(go.Bar(x=monthly['Month'], y=monthly['Debit'], name='Expense', marker_color='red'))
    fig2.update_layout(barmode='group', xaxis_title='Month', yaxis_title='Amount (₹)')
    st.plotly_chart(fig2, use_container_width=True)

    # --- Line Chart: Net Flow ---
    st.write("#### Net Flow Over Time")
    data['Net Flow'] = data['Credit'] - data['Debit']
    net_cumsum = data.groupby('Date')['Net Flow'].sum().cumsum().reset_index()
    fig3 = px.line(net_cumsum, x='Date', y='Net Flow', title='Cumulative Net Flow')
    st.plotly_chart(fig3, use_container_width=True)

    # --- Forecasting with Prophet ---
    st.write("#### Forecasting Future Expenses (Prophet)")
    forecast_df = data.groupby('Date')['Debit'].sum().reset_index()
    forecast_df = forecast_df.rename(columns={'Date':'ds', 'Debit':'y'})
    if len(forecast_df) > 10:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False)
        m.fit(forecast_df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        fig4 = plot_plotly(m, forecast)
        st.plotly_chart(fig4, use_container_width=True)
        st.info("Forecast is for daily expenses. Use with caution; actuals may vary.")
    else:
        st.warning("Not enough data for forecasting.")

    # --- Anomaly Detection ---
    st.write("#### Anomaly Detection (High Expense Days)")
    q = forecast_df['y'].quantile(0.99)
    anomalies = forecast_df[forecast_df['y'] > q]
    if not anomalies.empty:
        st.error(f"{len(anomalies)} high-expense days detected:")
        st.dataframe(anomalies)
    else:
        st.success("No major anomalies detected.")

    # --- Downloadable Results ---
    st.write("#### Download Processed Data")
    st.markdown(get_table_download_link(data, filename="processed_bank_statement.csv"), unsafe_allow_html=True)

    st.caption("Made with ❤️ for Indian bank statements. Handles SBI, ICICI, PNB, APGB, and more!")

    # --- Recommendations & Insights (Expense Categories Only) ---
    st.markdown("## 📝 Recommendations & Insights")

    # Only consider expenses for top spending categories
    if 'Debit' in data.columns and data['Debit'].sum() > 0:
        expense_data = data[data['Debit'] > 0]
        top_cats = (
            expense_data.groupby('category')['Debit']
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
    elif 'Credit' in data.columns and 'Debit' in data.columns:
        # Fallback: use net flow if Debit is not available
        expense_data = data[(data['Credit'] - data['Debit']) < 0]
        top_cats = (
            expense_data.groupby('category')['Debit']
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(3)
        )
    elif amount_col in data.columns:
        expense_data = data[data[amount_col] < 0]
        top_cats = (
            expense_data.groupby('category')[amount_col]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(3)
        )
    else:
        top_cats = pd.Series(dtype=float)

    if not top_cats.empty:
        st.info(f"Your top spending categories are: {', '.join(top_cats.index)}. Consider reviewing these for savings opportunities.")
    else:
        st.info("No spending categories found.")

    # The rest of your insights (bank, savings rate, etc.) can follow here...
    top_banks = data.groupby('bank')['Credit'].sum().sort_values(ascending=False).head(1) if 'bank' in data.columns and 'Credit' in data.columns else None
    if top_banks is not None and not top_banks.empty:
        st.info(f"Your highest net flow is with: {top_banks.index[0]}.")

    savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0
    if savings_rate < 20:
        st.warning("⚠ Your savings rate is below 20%. Consider increasing your savings for better financial health.")
    else:
        st.success("🎉 Your savings rate is healthy!") 
   
    
