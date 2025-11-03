# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Data Visualization Dashboard", layout="wide")

st.title("📊 Data Visualization Dashboard")
st.write("Visualize key insights from your dataset located in the same folder.")

# --- Load Dataset Function ---
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# --- Load your dataset automatically ---
dataset_path = "cleaned_FIXED_indexed_data_20251027_202233.csv"

try:
    df = load_data(dataset_path)
    st.success(f"Loaded '{dataset_path}' successfully!")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# --- Dataset Preview ---
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# --- Summary Statistics ---
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# --- Select Columns for Visualization ---
st.sidebar.header("Visualization Options")
key_columns = st.sidebar.multiselect(
    "Select columns to visualize:",
    options=df.columns.tolist(),
    default=df.select_dtypes(include=np.number).columns.tolist()[:3]
)

if len(key_columns) == 0:
    st.warning("Please select at least one column to visualize.")
    st.stop()

# --- 1️⃣ Heatmap ---
st.subheader("🔥 Correlation Heatmap")
numeric_df = df[key_columns].select_dtypes(include=np.number)

if not numeric_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("No numeric columns selected for heatmap.")

# --- 2️⃣ Line Charts ---
st.subheader("📉 Line Chart(s)")
for col in key_columns:
    st.write(f"**{col}**")
    st.line_chart(df[col])

# --- 3️⃣ Histograms ---
st.subheader("📊 Histograms")
for col in key_columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info(f"Skipping '{col}' — not numeric.")

st.success("✅ Visualization complete!")
