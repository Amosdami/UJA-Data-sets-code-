# ============================================================
# STREAMLIT ELECTRICAL DATA DASHBOARD – CORRECTED & ROBUST
# - Explicit ID vs SMA source tagging
# - Validated measurement dictionary
# - Dictionary validation report panel
# - Safe plotting across mixed schemas
# - NO WARNINGS VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from datetime import datetime, timedelta, time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="UJA Electrical Data Dashboard", layout="wide")

# ============================================================
# MEASUREMENT DICTIONARY (SCHEMA-SAFE)
# ============================================================
MEASUREMENT_DICT = {
    "Voltage": {
        "AC": ["V_ID2", "V_ID3", "VAC_SMA"],
        "DC": ["V_ID1", "VDC_SMA"],
    },
    "Current": {
        "AC": ["I_ID2", "I_ID3", "I_SMA"],
        "DC": ["I_ID1", "IDC_SMA"],
    },
    "Power": {
        "AC": ["P_ID2", "P_ID3", "PAC_SMA"],
        "DC": ["Pot_ID1", "PDC_SMA"],
        "Reactive": ["Q_ID2", "Q_ID3"],
        "Apparent": ["S_ID2", "S_ID3"],
    },
    "Energy": {
        "AC": ["Etot_ID2", "Etot_ID3", "Ertot_ID2", "Ertot_ID3", "Etotal_SMA"],
        "DC": ["Eacumulada_ID1"],
    },
    "Frequency": {
        "AC": ["F_ID2", "F_SMA"],
    },
    "Power Factor": {
        "AC": ["Fdp_ID2", "Fdp_ID3"],
    },
    "Temperature": {
        "Sensor": ["T1_ID1", "T1_ID2", "Temp_SMA"],
    },
    "Solar Irradiance": {
        "Sensor": ["G_ID1"],
    },
}

# ============================================================
# UTILITIES
# ============================================================

def detect_source_type(columns):
    """Detect if data source is SMA inverter or ID-based sensor"""
    if any(c.endswith("_SMA") for c in columns):
        return "SMA"
    if any("_ID" in c for c in columns):
        return "ID"
    return "UNKNOWN"


def select_columns(df, measurement, subtype):
    """Returns all columns for the selected measurement that exist in df"""
    desired = MEASUREMENT_DICT[measurement][subtype]
    return [c for c in desired if c in df.columns]


def dictionary_validation(df):
    """Generate validation report showing expected vs found columns"""
    rows = []
    for meas, groups in MEASUREMENT_DICT.items():
        for group, cols in groups.items():
            found = [c for c in cols if c in df.columns]
            rows.append({
                "Measurement": meas,
                "Group": group,
                "Expected": ", ".join(cols),
                "Found": ", ".join(found) if found else "❌ None",
            })
    return pd.DataFrame(rows)


def parse_datetime_smart(fecha_str, hora_str):
    """Smart datetime parser that handles multiple formats without warnings"""
    combined = f"{fecha_str} {hora_str}"
    
    # Try common formats in order
    formats_to_try = [
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
    ]
    
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(combined, format=fmt)
        except:
            continue
    
    # If all fail, return NaT
    return pd.NaT

# ============================================================
# DATA LOADING
# ============================================================
BASE_DATA_DIR = r"C:/Users/23480/Desktop/streamlit_dashboard"

@st.cache_data
def load_data():
    """Load all CSV files from subdirectories with robust date parsing"""
    dfs = []
    for folder in os.listdir(BASE_DATA_DIR):
        folder_path = os.path.join(BASE_DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in glob.glob(os.path.join(folder_path, "*.csv")):
            try:
                df = pd.read_csv(file)
                if len(df.columns) == 1:
                    df = pd.read_csv(file, sep=';')
            except Exception:
                continue

            # Check for required columns (case-insensitive)
            fecha_col = next((c for c in df.columns if c.lower() == "fecha"), None)
            hora_col = next((c for c in df.columns if c.lower() == "hora"), None)
            
            if not fecha_col or not hora_col:
                continue

            # Smart datetime parsing without warnings
            df["DateTime"] = df.apply(
                lambda row: parse_datetime_smart(str(row[fecha_col]), str(row[hora_col])),
                axis=1
            )
            
            df = df.dropna(subset=["DateTime"])
            df = df.sort_values("DateTime")

            df["Source_House"] = folder
            df["Source_Type"] = detect_source_type(df.columns)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ============================================================
# LOAD DATA
# ============================================================
st.title("⚡ Electrical Data Dashboard (ID & SMA Aware)")

df = load_data()
if df.empty:
    st.error("No valid data found")
    st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar:
    st.header("🔧 Filters")
    houses = sorted(df["Source_House"].unique())
    selected_houses = st.multiselect("House", houses, default=houses)
    df = df[df["Source_House"].isin(selected_houses)]

    date_range = st.date_input("Date range", [df.DateTime.min().date(), df.DateTime.max().date()])
    
    # Handle single date or date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = date_range
    
    df = df[(df.DateTime.dt.date >= start_date) & (df.DateTime.dt.date <= end_date)]
    
    st.info(f"📊 Filtered: {len(df):,} rows")

# ============================================================
# DICTIONARY VALIDATION PANEL
# ============================================================
with st.expander("🧪 Dictionary Validation Report", expanded=False):
    st.markdown("**This report shows which columns from the measurement dictionary exist in your current dataset:**")
    report = dictionary_validation(df)
    st.dataframe(report, width='stretch')

# Show all available columns for debugging
with st.expander("🔍 Debug: All Available Columns in Current Dataset", expanded=False):
    all_cols = sorted([col for col in df.columns if col not in ['DateTime', 'Source_House', 'Source_Type']])
    st.markdown(f"**Total columns: {len(all_cols)}**")
    
    # Separate ID and SMA columns
    id_cols = [c for c in all_cols if '_ID' in c]
    sma_cols = [c for c in all_cols if '_SMA' in c]
    other_cols = [c for c in all_cols if '_ID' not in c and '_SMA' not in c]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**ID Columns:**")
        st.write(id_cols if id_cols else "None")
    with col2:
        st.markdown("**SMA Columns:**")
        st.write(sma_cols if sma_cols else "None")
    with col3:
        st.markdown("**Other Columns:**")
        st.write(other_cols if other_cols else "None")

# ============================================================
# MEASUREMENT SELECTION
# ============================================================
st.markdown("### 📊 Select Measurement to Visualize")

measurement = st.selectbox("Measurement Type", list(MEASUREMENT_DICT.keys()))

# Only show types that have available data
types_available = [
    t for t in MEASUREMENT_DICT[measurement]
    if any(c in df.columns for c in MEASUREMENT_DICT[measurement][t])
]

if not types_available:
    st.warning(f"⚠️ No data available for {measurement} in current dataset.")
    st.stop()

subtype = st.radio("Type", types_available, horizontal=True)
columns = select_columns(df, measurement, subtype)

if not columns:
    st.warning("No columns available for this selection")
    st.stop()

st.success(f"✅ Plotting {len(columns)} columns: {', '.join(columns)}")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPIs", 
    "📈 Time Series", 
    "📊 Histogram", 
    "🔵 Scatter Plots",
    "🥧 Pie Chart"
])

# ============================================================
# TAB 1 – KPIs
# ============================================================
with tab1:
    st.subheader("Key Performance Indicators")
    
    for col in columns:
        st.markdown(f"#### 🔹 {col}")
        col_data = df[col].dropna()
        if col_data.empty:
            st.warning(f"No data for {col}")
            continue
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{col_data.mean():.2f}")
        c2.metric("Min", f"{col_data.min():.2f}")
        c3.metric("Max", f"{col_data.max():.2f}")
        c4.metric("Sum", f"{col_data.sum():.2f}")
        st.markdown("---")

# ============================================================
# TAB 2 – TIME SERIES
# ============================================================
with tab2:
    st.subheader(f"{measurement} ({subtype}) – Time Series")
    
    fig = go.Figure()
    for col in columns:
        temp = df[["DateTime", col]].dropna()
        if temp.empty:
            continue
        temp = temp.set_index("DateTime")[col].resample("5min").mean().reset_index()
        fig.add_trace(go.Scatter(
            x=temp.DateTime, 
            y=temp[col], 
            mode="lines+markers", 
            name=col,
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        height=500, 
        template="plotly_white", 
        hovermode="x unified",
        xaxis_title="DateTime",
        yaxis_title=measurement
    )
    st.plotly_chart(fig, width='stretch')

# ============================================================
# TAB 3 – HISTOGRAM
# ============================================================
with tab3:
    st.subheader(f"{measurement} ({subtype}) – Distribution Histogram")
    
    fig_hist = go.Figure()
    for col in columns:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        
        fig_hist.add_trace(
            go.Histogram(
                x=col_data,
                name=col,
                nbinsx=30,
                opacity=0.7
            )
        )
    
    fig_hist.update_layout(
        height=500,
        template="plotly_white",
        barmode='overlay',
        xaxis_title=measurement,
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_hist, width='stretch')

# ============================================================
# TAB 4 – SCATTER PLOTS
# ============================================================
with tab4:
    st.subheader(f"{measurement} ({subtype}) – Scatter Plots Over Time")
    
    for col in columns:
        st.markdown(f"### 🔹 Scatter Plot for: `{col}`")
        
        temp = df[['DateTime', col]].dropna()
        if temp.empty:
            st.warning(f"No data for {col}")
            continue
        
        fig_scatter = px.scatter(
            temp,
            x='DateTime',
            y=col,
            title=f"{col} Over Time",
            color_discrete_sequence=["#1f77b4"]
        )
        fig_scatter.update_traces(mode='markers', marker=dict(size=6))
        fig_scatter.update_layout(
            height=400,
            template='plotly_white',
            xaxis_title='DateTime',
            yaxis_title=col,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_scatter, width='stretch')

# ============================================================
# TAB 5 – PIE CHART
# ============================================================
with tab5:
    st.subheader(f"{measurement} ({subtype}) – Total Contribution by Column")
    
    # Calculate total sum for each column
    totals = []
    for col in columns:
        col_sum = df[col].sum()
        if col_sum > 0:
            totals.append({"Column": col, "Total": col_sum})
    
    if not totals:
        st.warning("No positive values found")
    else:
        df_totals = pd.DataFrame(totals)
        
        fig_pie = px.pie(
            df_totals,
            values="Total",
            names="Column",
            title=f"{measurement} ({subtype}) – Total Contribution",
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_pie, width='stretch')