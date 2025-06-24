import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from prophet import Prophet

st.set_page_config(page_title="Financial Insights Dashboard", layout="wide")
st.title("📊 Financial Insights & AI Predictions")

# File uploader for CSVs
uploaded_files = st.file_uploader(
    "Upload one or more bank statement CSV files",
    type="csv",
    accept_multiple_files=True
)

data = None
if uploaded_files:
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
    st.warning("Please upload one or more CSV files to begin.")

if data is not None and not data.empty:
    # 2. Data Cleaning
    data.columns = [col.lower().strip().replace(' ', '_') for col in data.columns]
    possible_date_cols = [col for col in data.columns if 'date' in col]
    possible_amount_cols = [col for col in data.columns if 'amount' in col or 'withdrawal' in col or 'deposit' in col]
    possible_balance_cols = [col for col in data.columns if 'balance' in col]
    DATE_COL = possible_date_cols[0] if possible_date_cols else data.columns[0]
    AMOUNT_COL = possible_amount_cols[0] if possible_amount_cols else data.columns[1]
    data[DATE_COL] = pd.to_datetime(data[DATE_COL], errors='coerce')
    data = data.dropna(subset=[DATE_COL])
    data[AMOUNT_COL] = pd.to_numeric(data[AMOUNT_COL], errors='coerce')
    data = data.dropna(subset=[AMOUNT_COL])
    data['month'] = data[DATE_COL].dt.to_period('M')

    # 3. Category Analysis
    def categorize(desc):
        desc = str(desc).lower()
        if 'salary' in desc or 'credit' in desc or 'neft' in desc: return 'Salary/Income'
        if 'grocery' in desc or 'supermarket' in desc or 'mart' in desc: return 'Groceries'
        if 'electric' in desc or 'water' in desc or 'gas' in desc or 'utility' in desc: return 'Utilities'
        if 'rent' in desc or 'lease' in desc: return 'Rent'
        if 'atm' in desc or 'cash' in desc: return 'Cash Withdrawal'
        if 'restaurant' in desc or 'food' in desc or 'cafe' in desc: return 'Food & Dining'
        if 'travel' in desc or 'uber' in desc or 'ola' in desc or 'flight' in desc: return 'Travel'
        if 'insurance' in desc: return 'Insurance'
        if 'emi' in desc or 'loan' in desc: return 'Loan/EMI'
        return 'Others'
    desc_col = [col for col in data.columns if 'desc' in col or 'particular' in col or 'narration' in col]
    if desc_col:
        data['category'] = data[desc_col[0]].apply(categorize)
    else:
        data['category'] = 'Others'

    # 4. Bank Extraction
    data['bank'] = data['source_file'].str.extract(r'(apgb|icici|pnb|sbi)', expand=False).str.upper().fillna('OTHER')

    # 5. Summary Stats
    st.header("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Total Net Flow", f"{data[AMOUNT_COL].sum():,.2f}")
    with col3:
        st.metric("Average Monthly Net Flow", f"{data.groupby('month')[AMOUNT_COL].sum().mean():,.2f}")

    # 6. Monthly Net Flow Chart
    st.subheader("Monthly Net Flow (Income - Expenses)")
    monthly = data.groupby('month')[AMOUNT_COL].sum()
    fig, ax = plt.subplots(figsize=(10,4))
    monthly.plot(kind='bar', ax=ax, color='skyblue')
    plt.ylabel('Net Amount')
    st.pyplot(fig)

    # 7. Category Spending Chart
    st.subheader("Monthly Spending by Category")
    cat_monthly = data.groupby(['month', 'category'])[AMOUNT_COL].sum().unstack().fillna(0)
    fig2, ax2 = plt.subplots(figsize=(12,5))
    cat_monthly.plot(kind='bar', stacked=True, ax=ax2)
    plt.ylabel('Amount')
    st.pyplot(fig2)

    # 8. Bank-wise Comparison
    st.subheader("Monthly Net Flow by Bank")
    bank_monthly = data.groupby(['month', 'bank'])[AMOUNT_COL].sum().unstack().fillna(0)
    fig3, ax3 = plt.subplots(figsize=(12,5))
    bank_monthly.plot(ax=ax3)
    plt.ylabel('Net Flow')
    st.pyplot(fig3)

    # 9. Prophet Forecast
    st.subheader("Net Flow Forecast (Next 6 Months)")
    df_prophet = monthly.reset_index().rename(columns={'month': 'ds', AMOUNT_COL: 'y'})
    df_prophet['ds'] = df_prophet['ds'].astype(str)
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)
    fig4 = m.plot(forecast)
    st.pyplot(fig4)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

    # 10. Top Expenses/Incomes
    st.subheader("Top 5 Largest Expenses")
    expenses = data[data[AMOUNT_COL] < 0].nsmallest(5, AMOUNT_COL)
    st.dataframe(expenses[[DATE_COL, AMOUNT_COL, 'category', 'source_file']])
    st.subheader("Top 5 Largest Incomes")
    incomes = data[data[AMOUNT_COL] > 0].nlargest(5, AMOUNT_COL)
    st.dataframe(incomes[[DATE_COL, AMOUNT_COL, 'category', 'source_file']])

    # 11. Anomaly Detection
    st.subheader("Anomalous Transactions (Potential Outliers)")
    iso = IsolationForest(contamination=0.01, random_state=42)
    data['anomaly'] = iso.fit_predict(data[[AMOUNT_COL]])
    anomalies = data[data['anomaly'] == -1]
    st.dataframe(anomalies[[DATE_COL, AMOUNT_COL, 'category', 'source_file']].head(10))

    # 12. Recommendations
    st.header("Recommendations & Insights")
    next_month_pred = forecast['yhat'].iloc[-1]
    if next_month_pred < 0:
        st.warning('Your predicted net flow for next month is negative. Consider reducing discretionary expenses or increasing income sources!')
    else:
        st.success('Your predicted net flow for next month is positive. Keep up the good financial habits!')
    if len(anomalies) > 0:
        st.info('Review the anomalous transactions for possible errors or fraud.')
    top_cats = cat_monthly.sum().sort_values(ascending=False).head(3)
    st.write(f"Your top spending categories are: {', '.join(top_cats.index)}. Consider reviewing these for savings opportunities.")
    top_banks = bank_monthly.sum().sort_values(ascending=False).head(1)
    st.write(f"Your highest net flow is with: {top_banks.index[0]}.")
else:
    st.warning("No data loaded. Please ensure your CSV files are in the same directory as this app.") 