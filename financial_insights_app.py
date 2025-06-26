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
    amount_col = st.selectbox("Select the amount column", options=data.columns)
    type_col = st.selectbox("Select the type column (Credit/Debit, optional)", options=["None"] + list(data.columns))

    # --- Parse Dates ---
    data['Date'] = pd.to_datetime(data[date_col], errors='coerce')
    if desc_col != "None":
        data['Description'] = data[desc_col].astype(str)
    else:
        data['Description'] = ""

    # --- Determine Credit and Debit using 'Type' column if present ---
    if type_col != "None" and type_col in data.columns:
        data['Type'] = data[type_col].str.lower()
        data['Credit'] = np.where(data['Type'] == 'credit', data[amount_col], 0)
        data['Debit'] = np.where(data['Type'] == 'debit', data[amount_col], 0)
    else:
        data['Credit'] = np.where(data[amount_col] > 0, data[amount_col], 0)
        data['Debit'] = np.where(data[amount_col] < 0, -data[amount_col], 0)

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
            if 'food' in desc or 'restaurant' in desc or 'dining' in desc or 'cafe' in desc:
                return 'Food & Dining'
            if 'travel' in desc or 'flight' in desc or 'train' in desc or 'uber' in desc or 'ola' in desc:
                return 'Travel'
            if 'shopping' in desc or 'amazon' in desc or 'flipkart' in desc:
                return 'Shopping'
            if 'medical' in desc or 'hospital' in desc or 'pharmacy' in desc:
                return 'Medical'
            if 'insurance' in desc:
                return 'Insurance'
            if 'emi' in desc or 'loan' in desc:
                return 'Loan/EMI'
            if 'fuel' in desc or 'petrol' in desc or 'diesel' in desc:
                return 'Fuel'
            return 'Other Expense'

    data['Net'] = data['Credit'] - data['Debit']
    data['category'] = data.apply(lambda row: categorize(row['Description'], row['Net']), axis=1)

    # --- Main Tabs ---
    tab1, tab2, tab3 = st.tabs(["Overview", "Insights", "Prediction"])

    with tab1:
        st.markdown("### 📅 Monthly Summary")
        data['month'] = data['Date'].dt.to_period('M').astype(str)
        monthly = data.groupby('month').agg({'Credit': 'sum', 'Debit': 'sum', 'Net': 'sum'})
        st.dataframe(monthly)

        st.markdown("### 🥧 Expense Breakdown")
        # Only show true expenses (exclude income categories)
        expense_cats = data.loc[~data['category'].isin(['Salary/Income', 'Other Income']) & (data['Debit'] > 0)]
        pie_data = expense_cats.groupby('category')['Debit'].sum()
        fig_pie, ax_pie = plt.subplots()
        if not pie_data.empty:
            pie_data.plot.pie(autopct='%1.1f%%', ax=ax_pie)
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)
        else:
            st.info("No expenses found to plot.")

    with tab2:
        st.markdown("### 📝 Recommendations & Insights")
        total_income = data['Credit'].sum()
        total_expense = data['Debit'].sum()
        net_savings = total_income - total_expense

        st.metric("Total Income", f"₹{total_income:,.2f}")
        st.metric("Total Expenses", f"₹{total_expense:,.2f}")
        st.metric("Net Savings", f"₹{net_savings:,.2f}")

        # Top 3 expense categories
        top_expense_cats = expense_cats.groupby('category')['Debit'].sum().sort_values(ascending=False).head(3)
        if not top_expense_cats.empty:
            st.info(f"Your top spending categories are: {', '.join(top_expense_cats.index)}. Consider reviewing these for savings opportunities.")

        # Additional, easy-to-understand insights
        st.markdown("## 📊 Additional Insights")
        if total_income > 0:
            spend_ratio = total_expense / total_income
            if spend_ratio > 1:
                st.warning(f"You're spending more than you earn! For every ₹1 you earn, you spend ₹{spend_ratio:.2f}.")
            elif spend_ratio > 0.8:
                st.info(f"You're spending about ₹{spend_ratio:.2f} for every ₹1 you earn. Try to save more if possible.")
            else:
                st.success(f"Good job! You're saving a healthy portion of your income.")

        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        st.metric("Savings Rate (%)", f"{savings_rate:.2f}")

        if savings_rate < 10:
            st.warning("Your savings rate is quite low. Try to increase your savings for better financial security.")
        elif savings_rate > 30:
            st.success("Excellent savings rate! Keep it up.")

    with tab3:
        st.markdown("### 🔮 Net Flow Forecast (Next 6 Months)")
        monthly = data.groupby('month')['Net'].sum()
        df_prophet = monthly.reset_index().rename(columns={'month': 'ds', 'Net': 'y'})
        df_prophet['ds'] = df_prophet['ds'].astype(str)
        # Prophet needs at least 2 data points
        if df_prophet['y'].count() < 2:
            st.warning("Not enough data to make a forecast. Please upload more data.")
        else:
            m = Prophet()
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=6, freq='M')
            forecast = m.predict(future)
            fig4 = m.plot(forecast)
            st.pyplot(fig4)
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))
            next_month_pred = forecast.iloc[-6]['yhat']
            last_month_actual = df_prophet['y'].iloc[-1]
            trend = "increase" if next_month_pred > last_month_actual else "decrease"
            st.info(f"Your net flow is forecasted to {trend} next month (Predicted: ₹{next_month_pred:,.2f}).")
            if next_month_pred < 0:
                st.warning("Your forecasted net flow is negative. Consider reducing expenses, increasing income, or setting up an emergency fund.")
            else:
                st.success("Your forecasted net flow is positive! Consider increasing your savings or investments, and plan for future goals.")
            st.caption("Forecasts are based on your historical data and may vary.")

else:
    st.info("👆 Please upload one or more CSV files to begin.")
