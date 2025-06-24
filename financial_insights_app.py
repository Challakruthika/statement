import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from prophet import Prophet

st.set_page_config(page_title="Financial Insights Dashboard", layout="wide")
st.title("📊 Financial Insights & AI Predictions")

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
