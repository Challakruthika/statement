import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime

# 1. Read and merge all CSV files
all_files = glob.glob(os.path.join(os.getcwd(), '*.csv'))

# List to hold DataFrames
dfs = []
for filename in all_files:
    try:
        df = pd.read_csv(filename)
        df['source_file'] = os.path.basename(filename)
        dfs.append(df)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Concatenate all data
raw_data = pd.concat(dfs, ignore_index=True)

# 2. Data Cleaning and Standardization
# Try to standardize column names
raw_data.columns = [col.lower().strip().replace(' ', '_') for col in raw_data.columns]

# Guess likely column names
possible_date_cols = [col for col in raw_data.columns if 'date' in col]
possible_amount_cols = [col for col in raw_data.columns if 'amount' in col or 'withdrawal' in col or 'deposit' in col]
possible_balance_cols = [col for col in raw_data.columns if 'balance' in col]

# Use first found or fallback
DATE_COL = possible_date_cols[0] if possible_date_cols else raw_data.columns[0]
AMOUNT_COL = possible_amount_cols[0] if possible_amount_cols else raw_data.columns[1]
BALANCE_COL = possible_balance_cols[0] if possible_balance_cols else None

# Parse dates
raw_data[DATE_COL] = pd.to_datetime(raw_data[DATE_COL], errors='coerce')
raw_data = raw_data.dropna(subset=[DATE_COL])

# Try to ensure amount is numeric
raw_data[AMOUNT_COL] = pd.to_numeric(raw_data[AMOUNT_COL], errors='coerce')
raw_data = raw_data.dropna(subset=[AMOUNT_COL])

# 3. Exploratory Data Analysis
raw_data['month'] = raw_data[DATE_COL].dt.to_period('M')
monthly = raw_data.groupby('month')[AMOUNT_COL].sum()

plt.figure(figsize=(10,5))
monthly.plot(kind='bar')
plt.title('Monthly Net Flow (Income - Expenses)')
plt.ylabel('Net Amount')
plt.tight_layout()
plt.savefig('monthly_net_flow.png')
plt.close()

# 4. AI Prediction: Predict next month's net flow
X = np.arange(len(monthly)).reshape(-1,1)
y = monthly.values
model = LinearRegression()
model.fit(X, y)
next_month_pred = model.predict([[len(monthly)]])[0]

# 5. Anomaly Detection (Unusual Transactions)
iso = IsolationForest(contamination=0.01, random_state=42)
raw_data['anomaly'] = iso.fit_predict(raw_data[[AMOUNT_COL]])
anomalies = raw_data[raw_data['anomaly'] == -1]

# 6. Output Insights
print('--- Financial Insights ---')
print(f'Total records: {len(raw_data)}')
print(f'Total net flow: {raw_data[AMOUNT_COL].sum():.2f}')
print(f'Average monthly net flow: {monthly.mean():.2f}')
print(f'Predicted next month net flow: {next_month_pred:.2f}')
print(f'Number of anomalous transactions: {len(anomalies)}')

# Top 5 largest expenses
expenses = raw_data[raw_data[AMOUNT_COL] < 0].nsmallest(5, AMOUNT_COL)
print('\nTop 5 largest expenses:')
print(expenses[[DATE_COL, AMOUNT_COL, 'source_file']])

# Top 5 largest incomes
incomes = raw_data[raw_data[AMOUNT_COL] > 0].nlargest(5, AMOUNT_COL)
print('\nTop 5 largest incomes:')
print(incomes[[DATE_COL, AMOUNT_COL, 'source_file']])

print('\nSee monthly_net_flow.png for a chart of your monthly net flow.')

# Recommendations
if next_month_pred < 0:
    print('Warning: Your predicted net flow for next month is negative. Consider reducing expenses or increasing income!')
else:
    print('Good job! Your predicted net flow for next month is positive.')

if len(anomalies) > 0:
    print('Review the anomalous transactions for possible errors or fraud.') 