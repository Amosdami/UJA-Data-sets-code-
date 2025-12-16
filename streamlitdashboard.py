# ============================================================
# STREAMLIT ELECTRICAL DATA DASHBOARD – INTEGRATED VERSION
# Dictionary-based measurement selector with original column names
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

# Add light gray background and style
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #F5F5F5;
        }
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #E0E0E0;
        }
        /* Headers and text */
        h1, h2, h3, h4, h5, h6, p {
            color: #111111;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; margin-bottom:0px;'>⚡ UJA Electrical Data Dashboard</h1>
<p style='text-align:center; font-size:18px; margin-top:-10px;'>Advanced Comparison & Analysis of Multi-House Electrical Data</p>
<hr style='margin-top:10px;'>
""", unsafe_allow_html=True)

# ============================================================
# MEASUREMENT DICTIONARY (Updated with ACTUAL column names from CSVs)
# ============================================================
MEASUREMENT_DICT = {
    "Voltage": {
        "AC": [
            "V_ID2",        # Voltage ID2 (AC)
            "V_ID3",        # Voltage ID3 (AC)
            "Vmuestr",      # AC Voltage sample
            "VAC_SMA"       # SMA AC Voltage
        ],
        "DC": [
            "V_ID1",        # Voltage ID1 (DC)
            "V",            # DC Voltage
            "VDC_SMA"       # SMA DC Voltage
        ]
    },
    "Current": {
        "AC": [
            "I_ID2",        # Current ID2 (AC)
            "I_ID3",        # Current ID3 (AC)
            "I2",           # AC Current phase 2
            "I3",           # AC Current phase 3
            "Imuestr2",     # AC Current sample phase 2
            "Imuestr3",     # AC Current sample phase 3
            "I_SMA"         # SMA AC Current
        ],
        "DC": [
            "I_ID1",        # Current ID1 (DC)
            "I",            # DC Current
            "I1",           # DC Current phase 1
            "Imuestr",      # DC Current sample
            "Imuestr1",     # DC Current sample phase 1
            "IDC_SMA"       # SMA DC Current
        ]
    },
    "Power": {
        "Active (AC)": [
            "P_ID2",        # Active Power ID2
            "P_ID3",        # Active Power ID3
            "P2",           # AC Power phase 2
            "P3",           # AC Power phase 3
            "Pmuestr2",     # AC Power sample phase 2
            "Pmuestr3",     # AC Power sample phase 3
            "PAC_SMA"       # SMA AC Power
        ],
        "Active (DC)": [
            "Pot_ID1",      # Power ID1 (DC)
            "P",            # DC Power
            "P1",           # DC Power phase 1
            "Pmuestr",      # DC Power sample
            "Pmuestr1",     # DC Power sample phase 1
            "PDC_SMA"       # SMA DC Power
        ],
        "Reactive": [
            "Q_ID2",        # Reactive Power ID2
            "Q_ID3"         # Reactive Power ID3
        ],
        "Apparent": [
            "S_ID2",        # Apparent Power ID2
            "S_ID3",        # Apparent Power ID3
            "S_SMA"         # SMA Apparent Power
        ]
    },
    "Energy": {
        "AC": [
            "Ertot_ID2",    # Reactive Total Energy ID2
            "Ertot_ID3",    # Reactive Total Energy ID3
            "Etot_ID2",     # Total Energy ID2
            "Etot_ID3",     # Total Energy ID3
            "Eacum"         # Accumulated Energy (AC)
        ],
        "DC": [
            "Eacumulada_ID1",  # Accumulated Energy ID1
            "Eacum1",       # Accumulated Energy phase 1
            "Etotal_SMA"    # SMA Total Energy
        ]
    },
    "Power Factor": {
        "AC": [
            "Fdp_ID2",      # Power Factor ID2
            "Fdp_ID3"       # Power Factor ID3
        ]
    },
    "Frequency": {
        "AC": [
            "F_ID2",        # Frequency ID2
            "F_SMA"         # SMA Frequency
        ]
    },
    "Temperature": {
        "Sensor": [
            "T1_ID1",       # Temperature ID1
            "T1_ID2",       # Temperature ID2
            "Temp_SMA"      # SMA Temperature
        ]
    },
    "Solar Irradiance": {
        "Sensor": [
            "G_ID1"         # Solar irradiance ID1
        ]
    }
}

# ============================================================
# CUSTOM COLOR SCHEME
# ============================================================
HOUSE_COLORS = {
    "House 1": "#FF8C00",  # Orange
    "House 2": "#000080"   # Navy Blue
}

# ============================================================
# COLUMN SELECTOR FUNCTION
# ============================================================
def select_columns_from_dict(df, measurement, acdc):
    """
    Returns only columns that:
    1. Exist in the dataframe
    2. Are defined in MEASUREMENT_DICT
    """
    desired_cols = MEASUREMENT_DICT.get(measurement, {}).get(acdc, [])
    return [c for c in desired_cols if c in df.columns]

# ============================================================
# DATA LOADING (Auto-discover houses from folders)
# ============================================================
BASE_DATA_DIR = r"C:/Users/23480/Desktop/streamlit_dashboard"

DATE_COL_FECHA = ["fecha"]
DATE_COL_HORA = ["hora"]

@st.cache_data
def load_data():
    all_dfs = []

    for folder in os.listdir(BASE_DATA_DIR):
        folder_path = os.path.join(BASE_DATA_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        # Load both _CLEANED.csv and _cleaned.csv files (case-insensitive)
        csv_files = glob.glob(os.path.join(folder_path, "*_CLEANED.csv"))
        csv_files += glob.glob(os.path.join(folder_path, "*_cleaned.csv"))
        
        # Remove duplicates in case filesystem is case-insensitive
        csv_files = list(set(csv_files))
        
        # If still no files found, try loading all .csv files
        if not csv_files:
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for file in csv_files:
            try:
                # Try reading with comma separator first
                df = pd.read_csv(file)
                
                # If only one column, try semicolon separator
                if len(df.columns) == 1:
                    df = pd.read_csv(file, sep=';')
                    
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
                continue

            # Detect Fecha / Hora
            fecha_col = next((c for c in df.columns if c.lower() in DATE_COL_FECHA), None)
            hora_col = next((c for c in df.columns if c.lower() in DATE_COL_HORA), None)

            if not fecha_col or not hora_col:
                st.warning(f"Missing fecha or hora in {file}")
                continue

            # Drop empty fecha or hora
            df = df.dropna(subset=[fecha_col, hora_col])
            df = df[df[fecha_col].astype(str).str.strip() != ""]
            df = df[df[hora_col].astype(str).str.strip() != ""]

            # Create DateTime with explicit format
            df["DateTime"] = pd.to_datetime(
                df[fecha_col].astype(str) + " " + df[hora_col].astype(str),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce"
            )

            # Drop invalid datetime rows
            df = df.dropna(subset=["DateTime"])
            df = df.sort_values("DateTime")

            # Use folder name as house identifier
            df["Source_House"] = folder

            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

# ============================================================
# SIDEBAR FILTERS
# ============================================================
with st.sidebar.expander("🔧 Filters & Settings", expanded=True):
    st.markdown("### 📅 Select Date Range")
    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=7)

    date_range = st.date_input(
        "Date Range:",
        value=(default_start, default_end),
        min_value=datetime(2022, 1, 1).date(),
        max_value=datetime(2026, 12, 31).date()
    )

    st.markdown("### ⏱ Time of Day Filter")
    time_range = st.slider("Hours", 0, 23, (0, 23))

# ============================================================
# LOAD DATA
# ============================================================
with st.spinner("Loading data..."):
    df_main, file_summary = load_data()

# Show file loading summary in sidebar
with st.sidebar.expander("📁 File Loading Summary", expanded=False):
    for msg in file_summary:
        st.text(msg)

if df_main.empty:
    st.error("No valid data found. Check file paths and file loading summary in sidebar.")
    st.stop()

# Sidebar info
st.sidebar.info(f"📊 Loaded {len(df_main):,} rows")
st.sidebar.info(f"📅 Date range: {df_main['DateTime'].min()} to {df_main['DateTime'].max()}")

# ============================================================
# FILTER DATA
# ============================================================
start_date, end_date = date_range if len(date_range) == 2 else (date_range[0], date_range[0])
start_h, end_h = time_range

start_time = time(start_h)
end_time = time(end_h, 59, 59)

filtered = df_main[
    (df_main['DateTime'].dt.date >= start_date) &
    (df_main['DateTime'].dt.date <= end_date) &
    (df_main['DateTime'].dt.time >= start_time) &
    (df_main['DateTime'].dt.time <= end_time)
].copy()

st.sidebar.info(f"🔎 Filtered: {len(filtered):,} rows")

if filtered.empty:
    st.warning("⚠️ No data matches your filters.")
    st.stop()

# Remove empty columns
filtered = filtered.dropna(axis=1, how='all')

# ============================================================
# MEASUREMENT SELECTOR (Dictionary-based)
# ============================================================
st.markdown("### 📊 Select Measurement to Visualize")

# Show available columns for debugging
with st.expander("🔍 Debug: Available Columns in Dataset"):
    numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
    st.write(f"Total numeric columns: {len(numeric_cols)}")
    st.write(numeric_cols)

measurement_type = st.selectbox(
    "Measurement Type",
    list(MEASUREMENT_DICT.keys())
)

# Get available AC/DC types for selected measurement (handle measurements with only one type)
available_types = list(MEASUREMENT_DICT[measurement_type].keys())

if len(available_types) == 1:
    ac_dc_type = available_types[0]
    st.info(f"📌 Only **{ac_dc_type}** available for {measurement_type}")
else:
    ac_dc_type = st.radio(
        "Select AC or DC",
        available_types,
        horizontal=True
    )

columns_to_plot = select_columns_from_dict(
    filtered,
    measurement_type,
    ac_dc_type
)

# Show what was found
st.info(f"📋 Columns defined in dictionary for {measurement_type} ({ac_dc_type}):")
st.write(MEASUREMENT_DICT[measurement_type][ac_dc_type])

if not columns_to_plot:
    st.warning("⚠️ No matching columns found in the dataset.")
    st.info("💡 The column names in your CSV don't match the dictionary. Please check the 'Available Columns' above and update the MEASUREMENT_DICT accordingly.")
    st.stop()

st.success(f"✅ Plotting {len(columns_to_plot)} columns:")
st.write(columns_to_plot)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 KPIs", 
    "📈 Time Series", 
    "📊 Histogram", 
    "🔵 Scatter Plots",
    "🥧 Pie Chart", 
    "🔍 Correlation & Features"
])

# ============================================================
# TAB 1 – KPIs
# ============================================================
with tab1:
    st.subheader("Key Performance Indicators")
    
    for col in columns_to_plot:
        st.markdown(f"### 📊 {col}")
        col1, col2 = st.columns(2)
        for idx, house in enumerate(sorted(filtered['Source_House'].unique())):
            house_data = filtered[filtered['Source_House'] == house][col].dropna()
            if house_data.empty:
                continue
            total_sum = house_data.sum()
            average = house_data.mean()
            maximum = house_data.max()
            minimum = house_data.min()
            target_col = col1 if idx == 0 else col2
            with target_col:
                st.markdown(f"#### 🏠 {house}")
                m1, m2 = st.columns(2)
                m3, m4 = st.columns(2)
                with m1: st.metric("Total Sum", f"{total_sum:,.2f}")
                with m2: st.metric("Average", f"{average:,.2f}")
                with m3: st.metric("Maximum", f"{maximum:,.2f}")
                with m4: st.metric("Minimum", f"{minimum:,.2f}")
                st.markdown("---")

# ============================================================
# TAB 2 – TIME SERIES (All columns on ONE chart)
# ============================================================
with tab2:
    st.subheader(f"{measurement_type} ({ac_dc_type}) – Combined View")

    temp = filtered[["DateTime"] + columns_to_plot].copy()
    temp = temp.dropna(subset=columns_to_plot, how="all")

    if temp.empty:
        st.warning("No data available.")
    else:
        fig = go.Figure()

        # Each column gets ONE line (not split by house)
        for col in columns_to_plot:
            col_data = temp[["DateTime", col]].dropna()
            if col_data.empty:
                continue

            # Resample for smoother visualization (5-min average)
            col_data = col_data.set_index("DateTime")
            resampled = col_data[col].resample("5min").mean().dropna().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=resampled["DateTime"],
                    y=resampled[col],
                    mode="lines+markers",
                    name=col,  # Just the column name
                    marker=dict(size=4)
                )
            )

        fig.update_layout(
            height=500,
            title=f"{measurement_type} ({ac_dc_type}) – All Columns (5-min Average)",
            xaxis_title="DateTime",
            yaxis_title=measurement_type,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 3 – HISTOGRAM (All columns on ONE chart)
# ============================================================
with tab3:
    st.subheader(f"{measurement_type} ({ac_dc_type}) – Distribution Histograms")
    
    fig = go.Figure()
    
    for col in columns_to_plot:
        col_data = filtered[col].dropna()
        if col_data.empty:
            continue
        
        fig.add_trace(
            go.Histogram(
                x=col_data,
                name=col,
                nbinsx=30,
                opacity=0.7
            )
        )
    
    fig.update_layout(
        height=500,
        title=f"{measurement_type} ({ac_dc_type}) – Distribution Comparison",
        xaxis_title=measurement_type,
        yaxis_title="Frequency",
        barmode='overlay',
        template="plotly_white",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 4 – SCATTER PLOTS
# ============================================================
with tab4:
    st.subheader("Scatter Plots Over Time")

    for col in columns_to_plot:
        st.markdown(f"### 🔹 Scatter Plot for: `{col}`")
        
        temp = filtered[['DateTime', col, 'Source_House']].dropna()
        if temp.empty:
            st.warning(f"No data for {col}")
            continue

        houses = sorted(temp['Source_House'].unique())
        cols_layout = st.columns(len(houses))

        for idx, house in enumerate(houses):
            house_data = temp[temp['Source_House'] == house].copy()
            if house_data.empty:
                continue

            fig = px.scatter(
                house_data,
                x='DateTime',
                y=col,
                title=f"{house} – {col} Over Time",
                color_discrete_sequence=[HOUSE_COLORS.get(house, "#888888")]
            )
            fig.update_traces(mode='markers')
            fig.update_layout(
                height=400,
                template='plotly_white',
                xaxis_title='DateTime',
                yaxis_title=col,
                hovermode='x unified'
            )

            with cols_layout[idx]:
                st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 5 – PIE CHART (All columns combined)
# ============================================================
with tab5:
    st.subheader(f"{measurement_type} ({ac_dc_type}) – Total Contribution by Column")

    # Calculate total sum for each column
    totals = []
    for col in columns_to_plot:
        col_sum = filtered[col].sum()
        if col_sum > 0:
            totals.append({"Column": col, "Total": col_sum})
    
    if not totals:
        st.warning("No positive values found")
    else:
        df_totals = pd.DataFrame(totals)
        
        fig = px.pie(
            df_totals,
            values="Total",
            names="Column",
            title=f"{measurement_type} ({ac_dc_type}) – Total Contribution by Column",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig, width="stretch")

# ============================================================
# TAB 6 – FEATURES + CORRELATION
# ============================================================
with tab6:
    st.subheader("Feature Engineering & Correlation Matrix")
    for house in filtered['Source_House'].unique():
        st.markdown(f"### 🔹 {house}")
        hdf = filtered[filtered['Source_House'] == house].copy()
        hdf['Hour'] = hdf['DateTime'].dt.hour
        hdf['DayOfWeek'] = hdf['DateTime'].dt.dayofweek
        hdf['Weekend'] = (hdf['DayOfWeek'] >= 5).astype(int)
        st.write("**Sample engineered features:**")
        st.dataframe(hdf[['DateTime','Hour','DayOfWeek','Weekend']].head(10))
        
        numeric_df = hdf.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            title=f"Correlation Matrix – {house}",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(height=600, template="plotly_white")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough numeric columns for correlation.")