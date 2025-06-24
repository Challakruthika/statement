# Financial Insights Project

## Overview
This project provides AI-powered financial analysis and predictions from your bank statement CSV files. It includes a Python script, a Jupyter notebook, and a modern Streamlit web app for interactive insights, visualizations, and recommendations.

## Features
- Automatic ingestion and cleaning of multiple bank statement CSVs
- Monthly net flow analysis (income vs. expenses)
- Category-wise spending analysis (auto-categorization)
- Bank-wise comparison
- Anomaly detection (unusual transactions)
- AI-powered predictions (next month and next 6 months)
- Export of results to Excel
- Interactive web dashboard (Streamlit)
- Actionable financial recommendations

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/Challakruthika/data_bank.git
cd data_bank
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Add Your Data
Place your bank statement CSV files in the project folder. **Do not upload sensitive data to GitHub!**

## Usage

### Run the Python Script
```sh
python financial_insights.py
```
- Prints insights and saves a chart as `monthly_net_flow.png`.

### Run the Jupyter Notebook
```sh
jupyter notebook
```
- Open `Financial_Insights_Analysis.ipynb` and run all cells.
- Exports results to `financial_insights_summary.xlsx`.

### Run the Streamlit Web App
```sh
streamlit run financial_insights_app.py
```
- Opens an interactive dashboard in your browser at `http://localhost:8501`.

## Deploying on Streamlit Cloud
1. Push your code (without sensitive data) to this GitHub repo.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click "New app", select your repo, and choose `financial_insights_app.py` as the entry point.
4. Click "Deploy". Your app will be live on a public URL!

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[Add your license here] 