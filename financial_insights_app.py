import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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
    elif type_col != "None":
        amount_col = st.selectbox("Select the amount column", options=data.columns)
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')
        type_map = {'DR': -1, 'DEBIT': -1, 'D': -1, 'CR': 1, 'CREDIT': 1, 'C': 1}
        data[type_col] = data[type_col].astype(str).str.upper().str.strip()
        data['net_amount'] = data[amount_col] * data[type_col].map(type_map).fillna(0)
        amount_col = 'net_amount'
    else:
        amount_col = st.selectbox("Select the amount column", options=data.columns)
        amount_sign = st.radio(
            "In your selected amount column, what do positive values mean?",
            ("Income/Credit", "Expense/Debit")
        )
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')
        if amount_sign == "Expense/Debit":
            data[amount_col] = -data[amount_col]

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.dropna(subset=[date_col])
    data = data.dropna(subset=[amount_col])
    data['month'] = data[date_col].dt.to_period('M')

    # --- Categorization ---
    if desc_col != "None":
        descs = data[desc_col].astype(str).fillna("")
        n_clusters = min(8, len(data))  # up to 8 clusters or number of rows
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        X = tfidf.fit_transform(descs)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        data['cluster_id'] = clusters
        # Get top 3 words for each cluster
        feature_names = tfidf.get_feature_names_out()
        cluster_labels = {}
        for i in range(n_clusters):
            idx = np.where(clusters == i)[0]
            if len(idx) == 0:
                cluster_labels[i] = f"Cluster {i+1}"
                continue
            cluster_tfidf = X[idx].mean(axis=0)
            top_indices = np.array(cluster_tfidf).ravel().argsort()[::-1][:3]
            top_words = [feature_names[j] for j in top_indices if cluster_tfidf[0, j] > 0]
            if top_words:
                label = ', '.join(top_words)
                cluster_labels[i] = f"{label}"
            else:
                cluster_labels[i] = f"Cluster {i+1}"
        data['category'] = data['cluster_id'].map(cluster_labels)
    else:
        data['category'] = 'Cluster 1'

    data['bank'] = data['source_file'].str.extract(r'(apgb|icici|pnb|sbi)', expand=False).str.upper().fillna('OTHER')

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
            savings_rate = 100 * (total_income - total_expense) / total_income if total_income != 0 else 0
        else:
            total_income = data[data[amount_col] > 0][amount_col].sum()
            total_expense = -data[data[amount_col] < 0][amount_col].sum()
            savings_rate = 100 * (total_income - total_expense) / total_income if total_income != 0 else 0

        with col4:
            st.metric("Savings Rate (%)", f"{savings_rate:.2f}")

        st.markdown("### 💵 Income vs. Expense Breakdown")
        st.success(f"Total Income: {total_income:,.2f}")
        st.error(f"Total Expenses: {total_expense:,.2f}")

        st.markdown("### 🥧 Expense Breakdown by Category")
        expense_data = data[(data[amount_col] < 0) & (~data['category'].isin(['Salary/Income', 'Other Income']))]
        expense_cats = expense_data.groupby('category')[amount_col].sum().abs().sort_values(ascending=False)
        if not expense_cats.empty:
            fig_pie, ax_pie = plt.subplots()
            expense_cats.plot.pie(autopct='%1.1f%%', ax=ax_pie, colormap='tab20')
            plt.ylabel('')
            st.pyplot(fig_pie)
            top3 = expense_cats.head(3)
            st.info(
                f"Top 3 Expense Categories:\n"
                + "\n".join([f"- {cat}: {amt:,.2f}" for cat, amt in top3.items()])
            )
        else:
            st.info("No expenses found for pie chart.")

        st.markdown("### 🏦 Bank-wise Net Flow")
        bank_total = data.groupby('bank')[amount_col].sum().sort_values(ascending=False)
        st.bar_chart(bank_total)
        if not bank_total.empty:
            top_bank = bank_total.index[0]
            st.info(f"Insight: Your highest net flow is with {top_bank} bank.")

        # Recommendations & Insights ONLY in tab1
        st.markdown("## 📝 Recommendations & Insights")
        top_cats = expense_cats.head(3)
        st.info(f"Your top spending categories are: {', '.join(top_cats.index)}. Consider reviewing these for savings opportunities.")
        top_banks = data.groupby('bank')[amount_col].sum().sort_values(ascending=False).head(1)
        st.info(f"Your highest net flow is with: {top_banks.index[0]}.")
        if savings_rate < 20:
            st.warning("⚠ Your savings rate is below 20%. Consider increasing your savings for better financial health.")
        else:
            st.success("🎉 Your savings rate is healthy!")

    with tab2:
        st.markdown("### 📅 Monthly Net Flow (Income - Expenses)")
        monthly = data.groupby('month')[amount_col].sum()
        fig, ax = plt.subplots(figsize=(10,4))
        monthly.plot(kind='bar', ax=ax, color='#4F8BF9')
        plt.ylabel('Net Amount')
        st.pyplot(fig)
        if not monthly.empty:
            trend = "increasing" if monthly.diff().mean() > 0 else "decreasing"
            best_month = monthly.idxmax()
            worst_month = monthly.idxmin()
            st.info(
                f"Trend: Your monthly net flow is {trend} over time.\n"
                f"Best Month: {best_month} ({monthly.max():,.2f})\n"
                f"Worst Month: {worst_month} ({monthly.min():,.2f})"
            )
            if (monthly < 0).any():
                st.warning("⚠ You had negative net flow in some months. Consider reviewing your expenses for those periods.")

        st.markdown("### 🏷 Monthly Spending by Category")
        cat_monthly = data.groupby(['month', 'category'])[amount_col].sum().unstack().fillna(0)
        fig2, ax2 = plt.subplots(figsize=(12,5))
        cat_monthly.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
        plt.ylabel('Amount')
        st.pyplot(fig2)
        top_cat = expense_cats.head(1)
        if not top_cat.empty:
            st.info(
                f"Insights:\n"
                f"- Your top spending category is {top_cat.index[0]} with a total of {top_cat.iloc[0]:,.2f}.\n"
                f"- Consider reviewing this category for potential savings."
            )

        st.markdown("### 🏦 Monthly Net Flow by Bank")
        bank_monthly = data.groupby(['month', 'bank'])[amount_col].sum().unstack().fillna(0)
        fig3, ax3 = plt.subplots(figsize=(12,5))
        bank_monthly.plot(ax=ax3)
        plt.ylabel('Net Flow')
        st.pyplot(fig3)

        # Additional Insights for Trends Tab
        st.markdown("## 📊 Trends Insights")
        st.info(f"Your net flow trend is {'increasing' if trend == 'increasing' else 'decreasing'}.")
        st.info(f"Best month: {best_month} ({monthly.max():,.2f}), Worst month: {worst_month} ({monthly.min():,.2f})")
        if not top_cat.empty:
            st.info(f"Top spending category: {top_cat.index[0]} ({top_cat.iloc[0]:,.2f})")

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
        st.write("• *yhat*: Predicted net amount (savings/overspending).")
        st.write("• *yhat_lower / yhat_upper*: Range of possible outcomes based on historical variability.")

        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

        next_month_pred = forecast['yhat'].iloc[-1]
        if next_month_pred < 0:
            st.warning(
                '📉 Your predicted net flow for next month is *negative*.\n\n'
                '👉 This means you might spend more than you earn. Consider reducing discretionary expenses or finding ways to increase income.'
            )
        else:
            st.success(
                '📈 Your predicted net flow for next month is *positive*.\n\n'
                '✅ Keep up your good financial habits and savings momentum!'
            )

        # Show summary of next 6 months forecast only once here
        st.markdown("### 📅 Summary of Next 6 Months Forecast")
        for i in range(1, 7):
            row = forecast.iloc[-i]
            label = pd.to_datetime(row['ds']).strftime('%B %Y')
            pred = row['yhat']
            if pred >= 0:
                st.markdown(f"✅ *{label}*: Projected Savings of ₹{pred:,.2f}")
            else:
                st.markdown(f"❌ *{label}*: Projected Overspending of ₹{-pred:,.2f}")

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

        st.markdown("### 🧠 AI Tip")
        if score < 25:
            st.info("Try identifying high-value non-essential expenses from your top 3 categories and set monthly caps for them.")
        elif score >= 25 and score < 50:
            st.info("Track your top expense categories weekly and move unspent amount into savings.")
        elif score >= 50:
            st.info("Automate savings by setting a recurring transfer to a high-interest savings account right after salary credit.")

        st.markdown("### 📊 Forecast-Based Insights")
        if next_month_pred < 0:
            st.error("🚨 Alert: Next month could result in overspending. Review subscriptions, food delivery, or shopping expenses.")
        else:
            st.success("✅ Forecast looks healthy. Continue monitoring and look for small consistent savings to improve score further.")

        st.markdown("### 💡 Suggestions")
        if next_month_pred < 0:
            st.markdown("- Review your top 3 spending categories and set monthly budgets for them.")
            st.markdown("- Consider pausing or cancelling unused subscriptions.")
            st.markdown("- Track your expenses weekly to avoid overspending.")
            st.markdown("- Explore ways to increase your income or savings rate.")
        else:
            st.markdown("- Consider increasing your monthly savings or investments.")
            st.markdown("- Review your expenses to see if you can save even more.")
            st.markdown("- Plan for future goals (travel, education, retirement) using your surplus.")
            st.markdown("- Continue monitoring your finances for consistent improvement.")

    with tab4:
        st.markdown("### 🚨 Anomalous Transactions (Potential Outliers)")
        iso = IsolationForest(contamination=0.01, random_state=42)
        data['anomaly'] = iso.fit_predict(data[[amount_col]])
        anomalies = data[data['anomaly'] == -1]
        st.dataframe(anomalies[[date_col, amount_col, 'category', 'source_file']].head(10))
        st.info(f"{len(anomalies)} anomalous transactions detected. Review these for possible errors or fraud.")

        # Additional Insights for Anomalies Tab
        st.markdown("## 🧐 Anomaly Insights")
        if len(anomalies) > 0:
            st.warning(f"{len(anomalies)} anomalies found. Review these transactions for possible errors, fraud, or unusual activity.")
            st.info("Tip: Large or unexpected transactions are often flagged as anomalies. Double-check these with your bank if unsure.")
        else:
            st.success("No significant anomalies detected in your transactions.")

    with tab5:
        st.markdown("### ⬇ Download Data & Forecast")
        st.download_button("Download Cleaned Data (CSV)", data.to_csv(index=False), "cleaned_data.csv")
        st.download_button("Download Forecast (CSV)", forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False), "forecast.csv")

st.markdown(
    "<hr style='margin-top:2em; margin-bottom:1em;'>"
    "<div style='text-align:center; color: #888;'>"
    "Chenna AI Tech Solutions "
    
    "</div>",
    unsafe_allow_html=True
)
      


 
    
