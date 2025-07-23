import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pulp import *
from datetime import datetime, date
import io
import zipfile
from ETL import etl_long_to_wide


# Page configuration
st.set_page_config(
    page_title="Amos - ENERGYNEST",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Thermal Storage Optimization System - DA Market")
st.markdown("""
This application optimizes thermal storage operations to minimize energy costs by:
- Charging thermal storage during low electricity prices
- Using stored energy during high prices
- Considering grid charges and market restrictions
""")

# Add helpful guidance
st.info("👈 **Getting Started:** Use the sidebar to upload your data file or configure API access, set system parameters, then run the optimization below.")

# Sidebar for parameters
st.sidebar.header("📁 Data Source")

# Add data source selection
data_source = st.sidebar.radio(
    "Select Data Source",
    ("Upload File", "Fetch from EnAppSys API", "Use Built-in EPEX 2024 Data"),
    help="Choose to upload a local CSV file, fetch data directly from EnAppSys API, or use the built-in EPEX 2024 dataset"
)

uploaded_file = None
api_config = None
use_builtin_data = False

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload electricity price data (CSV)", type=['csv'])
    
    # --- Existing transform data checkbox ---
    transform_data = st.sidebar.checkbox(
        "Transform price data (long to wide format)",
        value=True,
        help="Check this if your price data has one row per timestamp. Uncheck if your data is already wide (date, 00:00, 00:15...)."
    )
elif data_source == "Fetch from EnAppSys API":
    st.sidebar.subheader("🔌 EnAppSys API Configuration")
    
    # API Type Selection
    api_type = st.sidebar.selectbox(
        "API Type",
        ("chart", "bulk"),
        help="Chart API for specific chart codes, Bulk API for data types"
    )
    
    # Credentials
    with st.sidebar.expander("🔐 API Credentials", expanded=False):
        api_username = st.text_input("Username", type="default", help="Your EnAppSys username")
        api_password = st.text_input("Password", type="password", help="Your EnAppSys password")
    
    if api_type == "chart":
        # Chart API specific parameters
        chart_code = st.sidebar.text_input(
            "Chart Code", 
            value="de/elec/pricing/daprices",
            help="e.g., 'de/elec/pricing/daprices' for German day-ahead prices"
        )
        bulk_type = None
        entities = "ALL"
    else:  # bulk API
        # Bulk API specific parameters
        bulk_type = st.sidebar.selectbox(
            "Data Type",
            ("NL_SOLAR_FORECAST", "DE_WIND_FORECAST", "FR_DEMAND_FORECAST", "GB_DEMAND_FORECAST"),
            help="Select the type of bulk data to fetch"
        )
        entities = st.sidebar.text_input(
            "Entities", 
            value="ALL",
            help="Comma-separated list of entities or 'ALL' for all available"
        )
        chart_code = None
    
    # Common API parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        api_start_date = st.date_input(
            "Start Date", 
            value=pd.to_datetime("2024-01-01").date(),
            help="Start date for data fetch"
        )
    with col2:
        api_end_date = st.date_input(
            "End Date", 
            value=pd.to_datetime("2024-01-31").date(),
            help="End date for data fetch"
        )
    
    api_resolution = st.sidebar.selectbox(
        "Resolution",
        ("qh", "hourly", "daily", "weekly", "monthly"),
        index=0,
        help="Data resolution/frequency (qh = quarter hourly)"
    )
    
    api_timezone = st.sidebar.selectbox(
        "Timezone",
        ("CET", "WET", "EET", "UTC"),
        index=0,
        help="Timezone for the data"
    )
    
    api_currency = st.sidebar.selectbox(
        "Currency",
        ("EUR", "GBP", "USD"),
        index=0,
        help="Currency for price data"
    )
    
    # Validate credentials
    if not api_username or not api_password:
        st.sidebar.warning("⚠️ Please enter your API credentials to proceed")
        api_config = None
    else:
        # Build API config
        api_config = {
            'api_type': api_type,
            'credentials': {
                'user': api_username,
                'pass': api_password
            },
            'start_date': api_start_date.strftime('%d/%m/%Y 00:00'),
            'end_date': api_end_date.strftime('%d/%m/%Y 23:59'),
            'resolution': api_resolution,
            'timezone': api_timezone,
            'currency': api_currency
        }
        
        if api_type == "chart":
            api_config['chart_code'] = chart_code
        else:
            api_config['bulk_type'] = bulk_type
            api_config['entities'] = entities
    
    # For API data, we always want to transform (it comes in long format)
    transform_data = True
else:  # "Use Built-in EPEX 2024 Data"
    use_builtin_data = True
    st.sidebar.info("📊 Using built-in EPEX 2024 price data")
    st.sidebar.markdown("**Dataset:** `idprices-epex2024.csv`")
    # For built-in data, we always want to transform (it comes in long format)
    transform_data = True

st.sidebar.header("System Parameters")

# System parameters with defaults
Δt = st.sidebar.number_input("Time Interval (hours)", value=0.25, min_value=0.1, max_value=1.0, step=0.05)
Pmax_el = st.sidebar.number_input("Max Electrical Power (MW)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
Pmax_th = st.sidebar.number_input("Max Thermal Power (MW)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
Smax = st.sidebar.number_input("Max Storage Capacity (MWh)", value=8.0, min_value=1.0, max_value=50.0, step=0.5)
SOC_min = st.sidebar.number_input("Min Storage Level (MWh)", value=0.0, min_value=0.0, max_value=5.0, step=0.5)
η = st.sidebar.number_input("Charging Efficiency", value=0.95, min_value=0.7, max_value=1.0, step=0.05)
self_discharge_daily = st.sidebar.number_input("Self-Discharge Rate (% per day)", value=3.0, min_value=0.0 ,max_value=20.0, step=0.1, help="Daily percentage of stored energy lost due to standing thermal losses.")
C_grid = st.sidebar.number_input("Grid Charges (€/MWh)", value=30.0, min_value=0.0, max_value=100.0, step=1.0)
C_gas = st.sidebar.number_input("Gas Price (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)
boiler_efficiency_pct = st.sidebar.number_input( "Gas Boiler Efficiency (%)", value=90.0, min_value=50.0, max_value=100.0, step=1.0, help="Efficiency of the gas boiler in converting gas fuel to thermal energy.")
boiler_efficiency = boiler_efficiency_pct / 100.0
terminal_value = st.sidebar.number_input("Terminal Value (€/MWh)", value=65.0, min_value=10.0, max_value=200.0, step=1.0)

# --- NEW: DEMAND CONFIGURATION ---
st.sidebar.header("🔥 Thermal Demand Configuration")
demand_option = st.sidebar.radio(
    "Select Demand Source",
    ('Constant Demand', 'Upload Demand Profile'),
    help="Choose a fixed, constant demand or upload a CSV file with a time-varying demand profile."
)

D_th = None
demand_file = None
if demand_option == 'Constant Demand':
    D_th = st.sidebar.number_input("Thermal Demand (MW)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
else: # 'Upload Demand Profile'
    demand_file = st.sidebar.file_uploader("Upload customer demand data (CSV)", type=['csv'])

# --- UPDATED: Hochlast periods ---
st.sidebar.header("Peak Period Restrictions")
peak_period_option = st.sidebar.radio(
    "Define Peak Periods",
    ("Manual Selection", "Upload CSV File"),
    index=0,  # Default to Manual Selection
    help="Choose how to define peak restriction periods. Manual selection uses fixed checkboxes. Uploading a CSV allows for date-specific schedules."
)

hochlast_intervals_static = set()
peak_period_file = None

if peak_period_option == "Manual Selection":
    hochlast_morning = st.sidebar.checkbox("Morning Peak (8-10 AM)", value=True)
    hochlast_evening = st.sidebar.checkbox("Evening Peak (6-8 PM)", value=True)
    if hochlast_morning:
        hochlast_intervals_static.update(range(32, 40))  # 8-10 AM
    if hochlast_evening:
        hochlast_intervals_static.update(range(72, 80))  # 6-8 PM
else:  # "Upload CSV File"
    peak_period_file = st.sidebar.file_uploader(
        "Upload Peak Period CSV",
        type=['csv'],
        help="Upload a CSV with 'Date (CET)' and 'Is HLF' (1 for restricted, 0 for not)."
    )

# Holiday dates
with st.sidebar.expander("Holiday Configuration"):
    default_holidays = [
        '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-01', '2024-05-09',
        '2024-05-10', '2024-05-20', '2024-05-30', '2024-05-31', '2024-10-01',
        '2024-10-04', '2024-11-01', '2024-12-23', '2024-12-24', '2024-12-25',
        '2024-12-26', '2024-12-27', '2024-12-28', '2024-12-29', '2024-12-30',
        '2024-12-31'
    ]
    holiday_input = st.text_area("Holiday Dates (one per line, YYYY-MM-DD)",
                                 value='\n'.join(default_holidays),
                                 height=150)
    holiday_dates = [date.strip() for date in holiday_input.split('\n') if date.strip()]
    holiday_set = set(holiday_dates)

# Cache Management
with st.sidebar.expander("💾 Cache Management"):
    cached_items = []
    if 'cached_df_price' in st.session_state:
        cached_items.append("✅ Price Data")
    if 'cached_df_demand' in st.session_state:
        cached_items.append("✅ Demand Data")
    if 'cached_df_peak' in st.session_state:
        cached_items.append("✅ Peak Restriction Data")
    
    if cached_items:
        st.write("**Cached Data:**")
        for item in cached_items:
            st.write(item)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All Cache", help="Clear all cached data and force refresh"):
                # Clear all cache
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('cached_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("🔄 Refresh Data", help="Clear cache and reload current configuration"):
                # Clear all cache
                keys_to_remove = [k for k in st.session_state.keys() if k.startswith('cached_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.rerun()
    else:
        st.write("No cached data")
        st.caption("Data will be automatically cached after first fetch/upload")

# --- MAIN LOGIC CHANGE ---
if uploaded_file is not None or api_config is not None or use_builtin_data:
    
    # Create a unique key for the current configuration
    config_key = None
    if uploaded_file is not None:
        config_key = f"file_{uploaded_file.name}_{hash(uploaded_file.getvalue())}"
    elif api_config is not None:
        config_key = f"api_{hash(str(sorted(api_config.items())))}"
    elif use_builtin_data:
        config_key = "builtin_epex2024"
    
    # Check if we have cached data for this exact configuration
    if ('cached_config_key' in st.session_state and 
        'cached_df_price' in st.session_state and 
        st.session_state['cached_config_key'] == config_key):
        
        # Use cached data
        df_price = st.session_state['cached_df_price']
        st.success("✅ Using cached data from previous fetch!")
        
    else:
        # Need to fetch new data
        df_price = None

        if transform_data:
            st.info("New data source detected. Running ETL transformation...")
            with st.spinner("Transforming data from long to wide format..."):
                try:
                    if uploaded_file is not None:
                        # File upload mode
                        df_price = etl_long_to_wide(
                            input_source=uploaded_file,
                            datetime_column_name='Date (CET)',
                            value_column_name='Day Ahead Price'
                        )
                        st.success("✅ Price ETL transformation successful!")
                    elif api_config is not None:
                        # API mode
                        df_price = etl_long_to_wide(
                            use_api=True,
                            api_config=api_config
                        )
                        st.success("✅ API data fetched and ETL transformation successful!")
                    elif use_builtin_data:
                        # Built-in data mode
                        df_price = etl_long_to_wide(
                            input_source="idprices-epex2024.csv",
                            datetime_column_name='Date (CET)',
                            value_column_name='Day Ahead Price'
                        )
                        st.success("✅ Built-in EPEX 2024 data fetched and ETL transformation successful!")
                        
                    # Cache the successful result
                    if df_price is not None:
                        st.session_state['cached_df_price'] = df_price.copy()
                        st.session_state['cached_config_key'] = config_key
                        
                except Exception as e:
                    st.error(f"❌ ETL process failed: {e}")
                    st.info("Please check your configuration. For API: verify credentials and parameters. For files: ensure the header row is correct and column names match the expected input.")
                    st.stop()
        else:
            try:
                if uploaded_file is not None:
                    st.info("Loading price data directly in wide format.")
                    df_price = pd.read_csv(uploaded_file)
                elif use_builtin_data:
                    st.info("Loading price data directly in wide format from built-in EPEX 2024 data.")
                    df_price = pd.read_csv("idprices-epex2024.csv")
                else:
                    st.error("❌ API data must be transformed. Cannot load API data directly in wide format.")
                    st.stop()
                    
                # Cache the successful result
                if df_price is not None:
                    st.session_state['cached_df_price'] = df_price.copy()
                    st.session_state['cached_config_key'] = config_key
                    
            except Exception as e:
                st.error(f"❌ Failed to load the price CSV file: {e}")
                st.stop()

    # --- NEW: Process demand data if uploaded ---
    df_demand = None
    if demand_option == 'Upload Demand Profile':
        if demand_file is None:
            st.warning("Please upload a Customer Demand CSV file in the sidebar to proceed.")
            st.stop()
        else:
            # Create demand config key
            demand_config_key = f"demand_{demand_file.name}_{hash(demand_file.getvalue())}"
            
            # Check if we have cached demand data
            if ('cached_demand_config_key' in st.session_state and 
                'cached_df_demand' in st.session_state and 
                st.session_state['cached_demand_config_key'] == demand_config_key):
                
                # Use cached demand data
                df_demand = st.session_state['cached_df_demand']
                st.success("✅ Using cached demand data from previous upload!")
                
            else:
                # Need to process new demand data
                st.info("New demand file detected. Processing...")
                with st.spinner("Transforming demand data from long to wide format..."):
                    try:
                        df_demand = etl_long_to_wide(
                            input_source=demand_file,
                            datetime_column_name='Date (CET)',
                            value_column_name='MW-th'
                        )
                        st.success("✅ Demand ETL transformation successful!")
                        
                        # Cache the successful result
                        st.session_state['cached_df_demand'] = df_demand.copy()
                        st.session_state['cached_demand_config_key'] = demand_config_key
                        
                    except Exception as e:
                        st.error(f"❌ Demand file processing failed: {e}")
                        st.info("Please ensure the demand file has 'Date (CET)' and 'MW-th' columns.")
                        st.stop()

    # --- NEW: Process peak restriction data if uploaded ---
    df_peak = None
    if peak_period_option == 'Upload CSV File':
        if peak_period_file is None:
            st.warning("Please upload a Peak Period CSV file in the sidebar to proceed.")
            st.stop()
        else:
            # Create peak config key
            peak_config_key = f"peak_{peak_period_file.name}_{hash(peak_period_file.getvalue())}"
            
            # Check if we have cached peak data
            if ('cached_peak_config_key' in st.session_state and 
                'cached_df_peak' in st.session_state and 
                st.session_state['cached_peak_config_key'] == peak_config_key):
                
                # Use cached peak data
                df_peak = st.session_state['cached_df_peak']
                st.success("✅ Using cached peak restriction data from previous upload!")
                
            else:
                # Need to process new peak data
                st.info("New peak restriction file detected. Processing...")
                with st.spinner("Analyzing and cleaning peak restriction file..."):
                    try:
                        # --- ROBUST FILE HANDLING LOGIC ---
                        
                        # 1. Read the file to find the real header row
                        peak_period_file.seek(0) # Reset file pointer
                        lines = peak_period_file.read().decode('utf-8-sig').splitlines()
                        
                        header_row_index = -1
                        for i, line in enumerate(lines):
                            if 'Date (CET)' in line and 'Is HLF' in line:
                                header_row_index = i
                                break
                        
                        if header_row_index == -1:
                            st.error("❌ Invalid Peak Restriction File: Could not find the required header row with 'Date (CET)' and 'Is HLF' columns.")
                            st.stop()

                        # 2. Re-read the CSV from the correct line and pass to ETL
                        # We convert the clean part of the file into an in-memory text stream
                        clean_csv_in_memory = io.StringIO('\n'.join(lines[header_row_index:]))
                        
                        df_peak = etl_long_to_wide(
                            input_source=clean_csv_in_memory,
                            datetime_column_name='Date (CET)',
                            value_column_name='Is HLF'
                        )
                        st.success("✅ Peak restriction data cleaned and ETL successful!")
                        
                        # Cache the successful result
                        st.session_state['cached_df_peak'] = df_peak.copy()
                        st.session_state['cached_peak_config_key'] = peak_config_key

                    except Exception as e:
                        st.error(f"❌ A critical error occurred while processing the peak restriction file: {e}")
                        st.info("Please ensure the file contains 'Date (CET)' and 'Is HLF' columns somewhere inside it.")
                        st.stop()

    # Continue if price data is loaded
    if df_price is not None:
        try:
            st.sidebar.header("🗓️ Date Range Filter")
            df_price['date_obj'] = pd.to_datetime(df_price['date'])
            min_date = df_price['date_obj'].min().date()
            max_date = df_price['date_obj'].max().date()
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            if start_date > end_date:
                st.sidebar.error("Error: Start date cannot be after end date.")
                st.stop()

            mask = (df_price['date_obj'] >= pd.to_datetime(start_date)) & (df_price['date_obj'] <= pd.to_datetime(end_date))
            df_filtered = df_price.loc[mask].drop(columns=['date_obj'])

            # --- NEW: Merge price and demand dataframes ---
            df_processed = df_filtered
            if df_demand is not None:
                original_days = len(df_filtered)
                # Use an inner merge to only keep days with both price and demand data
                df_processed = pd.merge(df_filtered, df_demand, on='date', how='inner', suffixes=('_price', '_demand'))
                merged_days = len(df_processed)

                if merged_days == 0:
                    st.error("❌ No matching dates found between price data and demand data. Please check your files.")
                    st.stop()
                elif merged_days < original_days:
                    st.warning(f"⚠️ Found {merged_days} matching days. {original_days - merged_days} days were dropped due to no corresponding demand data.")
                else:
                    st.success(f"✅ Successfully merged price and demand data for {merged_days} days.")

            # --- UPDATED: Merge price, demand, and peak dataframes ---
            if df_peak is not None:
                original_rows = len(df_processed)
                
                # Clean debug: Show essential peak dataframe info
                peak_time_cols = [col for col in df_peak.columns if col != 'date']
                st.info(f"🔍 Peak data: {df_peak.shape[0]} days, {len(peak_time_cols)} time intervals")
                
                df_processed = pd.merge(df_processed, df_peak, on='date', how='left', suffixes=('', '_hlf'))
                
                # Find HLF columns and fill any missing days with 0 (not restricted)
                hlf_cols = [c for c in df_processed.columns if c.endswith('_hlf')]
                
                # If no _hlf columns found, check if peak columns were merged without suffix
                if not hlf_cols:
                    # Look for columns that exist in both original df_peak and merged df_processed (excluding 'date')
                    potential_hlf_cols = [col for col in df_peak.columns if col != 'date' and col in df_processed.columns]
                    if potential_hlf_cols:
                        hlf_cols = potential_hlf_cols
                
                if hlf_cols:
                    df_processed[hlf_cols] = df_processed[hlf_cols].fillna(0)
                    st.success(f"✅ Successfully merged peak restriction data. Found {len(hlf_cols)} time periods with restrictions.")
                    
                    # Show sample restriction stats only
                    sample_day = df_processed.iloc[0]
                    peak_values = sample_day[hlf_cols].values
                    restricted_count = sum(peak_values)
                    if restricted_count > 0:
                        st.info(f"📊 Sample: {restricted_count} out of {len(peak_values)} intervals restricted on {sample_day['date']}")
                else:
                    st.warning("⚠️ No peak restriction columns found after merging.")
                    # Detailed debug only for error cases
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Main data columns: {len(df_processed.columns)} total")
                        st.write(f"Peak data had: {len(df_peak.columns)} columns")
                        
                        # Check for date overlap
                        peak_dates = set(df_peak['date'].unique())
                        main_dates = set(df_processed['date'].unique())
                        overlap = peak_dates.intersection(main_dates)
                        st.write(f"Date overlap: {len(overlap)} out of {len(main_dates)} main dates have peak data")
                        
                        if len(overlap) == 0:
                            st.error("No overlapping dates found! Check date formats in your files.")
                        else:
                            st.write("Sample dates from peak data:", list(peak_dates)[:5])
                            st.write("Sample dates from main data:", list(main_dates)[:5])


            st.success(f"✅ Ready to analyze {len(df_processed)} days of data within the selected range ({start_date} to {end_date})")

            with st.expander("📊 Data Preview (filtered and merged)"):
                st.dataframe(df_processed.head(9))

            st.header("Data Cleaning")
            with st.spinner("Cleaning data..."):
                for col in df_processed.columns:
                    if col != 'date':
                        df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                        df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both')
                        if df_processed[col].isna().any():
                            col_median = df_processed[col].median()
                            df_processed[col] = df_processed[col].fillna(col_median)
            st.success("✅ Data cleaning completed")

            # --- NEW: Identify time columns based on demand option ---
            if df_demand is not None:
                price_time_cols = [col for col in df_processed.columns if col.endswith('_price')]
                demand_time_cols = [col for col in df_processed.columns if col.endswith('_demand')]
            else:
                price_time_cols = [col for col in df_processed.columns if col != 'date']
                demand_time_cols = [] # Not used for constant demand

            # --- NEW: Identify HLF time columns if peak restrictions are uploaded ---
            hlf_time_cols = []
            if df_peak is not None:
                # First try to find columns with _hlf suffix
                hlf_time_cols = [col for col in df_processed.columns if col.endswith('_hlf')]
                
                # If no _hlf suffix columns, look for the original time columns from peak data
                if not hlf_time_cols:
                    hlf_time_cols = [col for col in df_processed.columns if col in df_peak.columns and col != 'date']
                
                if hlf_time_cols:
                    st.success(f"✅ Ready to use {len(hlf_time_cols)} peak restriction time intervals.")
            else:
                st.info("ℹ️ Using static peak restrictions only.")


            # --- UPDATED: build_thermal_model to accept a demand profile ---
            def build_thermal_model(prices, demand_profile, soc0, η_self, boiler_eff, peak_restrictions=None, is_holiday=False):
                """Build optimization model for thermal storage system"""
                T = len(prices)
                model = LpProblem("Thermal_Storage", LpMinimize)

                p_el = LpVariable.dicts("p_el", range(T), lowBound=0, upBound=Pmax_el)
                p_th = LpVariable.dicts("p_th", range(T), lowBound=0, upBound=Pmax_th)
                p_gas = LpVariable.dicts("p_gas", range(T), lowBound=0)
                soc = LpVariable.dicts("soc", range(T), lowBound=SOC_min, upBound=Smax)

                model += lpSum([
                    (prices[t] + C_grid) * p_el[t] * Δt +
                    (C_gas / boiler_eff) * p_gas[t] * Δt
                    for t in range(T)
                ]) - terminal_value * soc[T-1]

                for t in range(T):
                    # --- UPDATED: Thermal balance uses the demand profile ---
                    model += p_th[t] + p_gas[t] == demand_profile[t]

                    # --- UPDATED: Handle both static and dynamic peak restrictions ---
                    if not is_holiday:
                        # Static restrictions (from manual selection)
                        if t in hochlast_intervals_static:
                            model += p_el[t] == 0
                        # Dynamic restrictions (from CSV file)
                        elif peak_restrictions is not None and len(peak_restrictions) > t and peak_restrictions[t] == 1:
                            model += p_el[t] == 0

                    if t == 0:
                        model += soc[t] == soc0 * η_self + η * p_el[t] * Δt - p_th[t] * Δt
                    else:
                        model += soc[t] == soc[t-1] * η_self + η * p_el[t] * Δt - p_th[t] * Δt

                return model, p_el, p_th, p_gas, soc

            if st.button("🚀 Run Optimization", type="primary"):
                if 'results' in st.session_state: del st.session_state['results']
                if 'all_trades' in st.session_state: del st.session_state['all_trades']
                if 'gas_baseline' in st.session_state: del st.session_state['gas_baseline']

                progress_bar = st.progress(0)
                status_text = st.empty()

                soc0 = SOC_min
                results = []
                all_trades = []
                all_baselines = []
                η_self = (1 - self_discharge_daily / 100) ** (Δt / 24)

                for idx, (_, row) in enumerate(df_processed.iterrows()):
                    progress_bar.progress((idx + 1) / len(df_processed))
                    day = row['date']
                    status_text.text(f"Processing day {idx + 1}/{len(df_processed)}: {day}")

                    prices = row[price_time_cols].values

                    # --- UPDATED: Determine demand profile for the day ---
                    if demand_option == 'Constant Demand':
                        demand_profile = np.full(len(prices), D_th)
                    else:
                        demand_profile = row[demand_time_cols].values

                    if len(prices) != len(demand_profile):
                        st.warning(f"Skipping day {day} due to mismatched data length.")
                        continue

                    # --- UPDATED: Daily gas baseline depends on the demand profile ---
                    gas_baseline_daily = (sum(demand_profile) * Δt * C_gas) / boiler_efficiency
                    all_baselines.append(gas_baseline_daily)

                    is_holiday = day in holiday_set

                    # --- UPDATED: Pass the day-specific peak restrictions to the model ---
                    peak_restrictions_for_day = None
                    if df_peak is not None and hlf_time_cols:
                        # Extract the HLF values for this specific day
                        day_row = df_processed[df_processed['date'] == day]
                        if not day_row.empty:
                            peak_restrictions_for_day = day_row[hlf_time_cols].iloc[0].values

                    model, p_el, p_th, p_gas, soc = build_thermal_model(prices, demand_profile, soc0, η_self, boiler_efficiency, peak_restrictions_for_day, is_holiday)
                    status = model.solve(PULP_CBC_CMD(msg=False))

                    if status == 1:
                        actual_elec_cost = sum((prices[t] + C_grid) * p_el[t].value() * Δt for t in range(len(prices)))
                        actual_gas_cost = sum(C_gas * (p_gas[t].value() / boiler_efficiency) * Δt for t in range(len(prices)))
                        soc_end = soc[len(prices)-1].value()

                        actual_total_cost = actual_elec_cost + actual_gas_cost - terminal_value * soc_end
                        savings = gas_baseline_daily - actual_total_cost

                        elec_energy = sum([p_el[t].value() * Δt for t in range(len(prices))])
                        gas_fuel_energy = sum([(p_gas[t].value() / boiler_efficiency) * Δt for t in range(len(prices))])

                        for t in range(len(prices)):
                            interval_hour, interval_min = divmod(t * 15, 60)
                            time_str = f"{interval_hour:02d}:{interval_min:02d}:00"
                            gas_cost_interval_val = C_gas * (p_gas[t].value() / boiler_efficiency) * Δt
                            elec_cost_interval_val = (prices[t] + C_grid) * p_el[t].value() * Δt

                            # Determine if this interval is restricted
                            is_static_restricted = t in hochlast_intervals_static and not is_holiday
                            is_dynamic_restricted = (peak_restrictions_for_day is not None and 
                                                   len(peak_restrictions_for_day) > t and 
                                                   peak_restrictions_for_day[t] == 1 and not is_holiday)
                            is_restricted = is_static_restricted or is_dynamic_restricted

                            trade_record = {
                                'date': day, 'time': time_str, 'interval': t, 'da_price': prices[t],
                                'total_elec_cost': prices[t] + C_grid, 'p_el_heater': p_el[t].value(),
                                'p_th_discharge': p_th[t].value(), 'p_gas_backup': p_gas[t].value(),
                                'soc': soc[t].value(), 'elec_cost_interval': elec_cost_interval_val,
                                'gas_cost_interval': gas_cost_interval_val,
                                'total_cost_interval': elec_cost_interval_val + gas_cost_interval_val,
                                'is_hochlast': is_restricted,
                                'is_holiday': is_holiday, 'is_charging': p_el[t].value() > 0.01,
                                'is_discharging': p_th[t].value() > 0.01, 'using_gas': p_gas[t].value() > 0.01,
                                'demand_th': demand_profile[t] # --- NEW: Store demand for plotting
                            }
                            all_trades.append(trade_record)
                        soc0 = soc_end
                        results.append({
                            "day": day, "cost": actual_total_cost, "savings": savings, "soc_end": soc_end,
                            "elec_energy": elec_energy, "gas_energy": gas_fuel_energy, "is_holiday": is_holiday,
                            "gas_baseline_daily": gas_baseline_daily
                        })

                progress_bar.progress(1.0)
                status_text.text("✅ Optimization completed!")

                st.session_state['results'] = results
                st.session_state['all_trades'] = all_trades
                st.session_state['gas_baseline'] = np.mean(all_baselines) # Store the average baseline

        except Exception as e:
            st.error(f"❌ An error occurred during optimization: {str(e)}")
            st.info("Please ensure your CSV files are in the expected format.")
            st.stop()

        # Display results
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            all_trades = st.session_state['all_trades']
            gas_baseline = st.session_state['gas_baseline']
            
            # Convert results to DataFrames for easier analysis
            results_df = pd.DataFrame(results)
            results_df['date'] = pd.to_datetime(results_df['day'])
            trades_df = pd.DataFrame(all_trades)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.header("📊 Results Summary")
            with col2:
                if st.button("🗑️ Clear Results"):
                    del st.session_state['results'], st.session_state['all_trades'], st.session_state['gas_baseline']
                    st.rerun()

            avg_cost = np.mean([r['cost'] for r in results])
            avg_savings = np.mean([r['savings'] for r in results])
            total_savings = sum([r['savings'] for r in results])
            avg_elec = np.mean([r['elec_energy'] for r in results])
            avg_gas = np.mean([r['gas_energy'] for r in results])
            savings_pct = (avg_savings / gas_baseline) * 100 if gas_baseline > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Days Analyzed", len(results))
            with col2:
                st.metric("Average Daily Savings", f"€{avg_savings:.0f}", f"{savings_pct:.1f}%")
            with col3:
                st.metric("Total Savings", f"€{total_savings:.0f}")
            with col4:
                # --- UPDATED: Metric label reflects the demand option ---
                baseline_label = "Avg. Gas Baseline/Day" if demand_option == 'Upload Demand Profile' else "Gas Baseline/Day"
                st.metric(baseline_label, f"€{gas_baseline:.0f}")

            thermal_from_elec = avg_elec * η
            thermal_from_gas = avg_gas * boiler_efficiency
            total_thermal_delivered = thermal_from_elec + thermal_from_gas
            elec_percentage, gas_percentage = (0,0)
            if total_thermal_delivered > 0:
                elec_percentage = (thermal_from_elec / total_thermal_delivered) * 100
                gas_percentage = (thermal_from_gas / total_thermal_delivered) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Thermal from Electricity", f"{elec_percentage:.1f}%")
            with col2:
                st.metric("Thermal from Gas", f"{gas_percentage:.1f}%")

            cost_gas_per_mwh_th = C_gas / boiler_efficiency
            break_even_price = (cost_gas_per_mwh_th * η) - C_grid
            st.info(f"**Break-even electricity price:** {break_even_price:.1f} €/MWh")

            best_day = max(results, key=lambda x: x['savings'])
            worst_day = min(results, key=lambda x: x['savings'])
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best day:** {best_day['day']} (€{best_day['savings']:.2f} saved)")
            with col2:
                st.warning(f"**Worst day:** {worst_day['day']} (€{worst_day['savings']:.2f} saved)")

            st.header("📈 Visualizations")
            
            fig1 = px.line(results_df, x='date', y='savings', title='Daily Savings Over Time', labels={'savings': 'Savings (€)', 'date': 'Date'})
            fig1.add_hline(y=avg_savings, line_dash="dash", annotation_text=f"Average: €{avg_savings:.2f}")
            st.plotly_chart(fig1, use_container_width=True)

            results_df['cumulative_savings'] = results_df['savings'].cumsum()
            fig2 = px.area(results_df, x='date', y='cumulative_savings', title='Cumulative Savings Over Time', labels={'cumulative_savings': 'Total Savings (€)', 'date': 'Date'})
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['elec_energy'], mode='lines', name='Electricity Input', fill='tonexty'))
            fig3.add_trace(go.Scatter(x=results_df['date'], y=results_df['gas_energy'], mode='lines', name='Gas Fuel Input', fill='tozeroy'))
            fig3.update_layout(title='Daily Energy Input Mix', xaxis_title='Date', yaxis_title='Energy (MWh)')
            st.plotly_chart(fig3, use_container_width=True)

            st.header("Sample Period Analysis")
            with st.expander("🔍 Detailed Period Analysis"):
                # --- MODIFICATION START ---
                trades_df['date_obj'] = pd.to_datetime(trades_df['date'])
                min_analysis_date = results_df['date'].min().date()
                max_analysis_date = results_df['date'].max().date()

                st.write("Select a date range for detailed operational analysis:")
                col1, col2 = st.columns(2)
                with col1:
                    start_date_analysis = st.date_input(
                        "Start Date",
                        value=min_analysis_date,
                        min_value=min_analysis_date,
                        max_value=max_analysis_date,
                        key="analysis_start"
                    )
                with col2:
                    # Auto-adjust end date if start date is later
                    default_end_date = max(min_analysis_date, start_date_analysis)
                    end_date_analysis = st.date_input(
                        "End Date",
                        value=default_end_date,
                        min_value=start_date_analysis,  # End date can't be before start date
                        max_value=max_analysis_date,
                        key="analysis_end"
                    )

                if start_date_analysis > end_date_analysis:
                    st.error("Analysis start date cannot be after the end date.")
                else:
                    mask = (trades_df['date_obj'].dt.date >= start_date_analysis) & (trades_df['date_obj'].dt.date <= end_date_analysis)
                    analysis_trades = trades_df[mask].copy()

                    if not analysis_trades.empty:
                        # Create a full datetime column for continuous plotting
                        analysis_trades['datetime'] = pd.to_datetime(analysis_trades['date'] + ' ' + analysis_trades['time'])
                        analysis_trades = analysis_trades.sort_values('datetime')

                        fig4 = make_subplots(rows=3, cols=1, subplot_titles=('Electricity Price & Storage Operations', 'State of Charge', 'Cost Breakdown'), vertical_spacing=0.1, specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])
                        
                        # Use the new 'datetime' column for the x-axis
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['da_price'], name='DA Price', line=dict(color='blue')), row=1, col=1, secondary_y=False)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_el_heater'], name='Charging', line=dict(color='green', dash='dot')), row=1, col=1, secondary_y=True)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['p_th_discharge'], name='Discharging', line=dict(color='red', dash='dot')), row=1, col=1, secondary_y=True)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['demand_th'], name='Thermal Demand', line=dict(color='purple', dash='longdash')), row=1, col=1, secondary_y=True)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['soc'], name='SOC', line=dict(color='orange')), row=2, col=1)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['elec_cost_interval'], name='Elec Cost', line=dict(color='blue')), row=3, col=1)
                        fig4.add_trace(go.Scatter(x=analysis_trades['datetime'], y=analysis_trades['gas_cost_interval'], name='Gas Cost', line=dict(color='red')), row=3, col=1)

                        fig4.update_layout(height=800, title_text=f"Detailed Analysis from {start_date_analysis} to {end_date_analysis}")
                        fig4.update_yaxes(title_text="Price (€/MWh)", row=1, col=1, secondary_y=False)
                        fig4.update_yaxes(title_text="Power (MW)", row=1, col=1, secondary_y=True, showgrid=False)
                        fig4.update_yaxes(title_text="Storage (MWh)", row=2, col=1)
                        fig4.update_yaxes(title_text="Cost (€)", row=3, col=1)
                        
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.warning("No data available for the selected date range.")
                # --- MODIFICATION END ---

            st.header("💾 Download Results")
            if not trades_df.empty:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    trades_csv = trades_df.to_csv(index=False)
                    zip_file.writestr('thermal_storage_trades.csv', trades_csv)
                    daily_csv = results_df.to_csv(index=False)
                    zip_file.writestr('thermal_storage_daily.csv', daily_csv)
                    params_text = f"""Thermal Storage Optimization Parameters
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Demand Option: {demand_option}
System Parameters:
- Time Interval: {Δt} hours
- Max Electrical Power: {Pmax_el} MW
- Max Thermal Power: {Pmax_th} MW
- Max Storage Capacity: {Smax} MWh
- Min Storage Level: {SOC_min} MWh
- Charging Efficiency: {η}
- Self-Discharge Rate: {self_discharge_daily} % per day
- Grid Charges: {C_grid} €/MWh
- Gas Price: {C_gas} €/MWh
- Gas Boiler Efficiency: {boiler_efficiency_pct} %
- Terminal Value: {terminal_value} €/MWh

Results Summary:
- Days Analyzed: {len(results)}
- Average Daily Savings: €{avg_savings:.2f} ({savings_pct:.1f}%)
- Total Savings: €{total_savings:.2f}
- Thermal Contribution from Electricity: {elec_percentage:.1f}%
- Break-even Price: {break_even_price:.1f} €/MWh
"""
                    zip_file.writestr('parameters_and_summary.txt', params_text)
                zip_buffer.seek(0)
                st.download_button(label="📥 Download All Results (ZIP)", data=zip_buffer.getvalue(), file_name=f"thermal_storage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(label="📊 Download Detailed Trades (CSV)", data=trades_df.to_csv(index=False), file_name='thermal_storage_trades.csv', mime='text/csv')
                with col2:
                    st.download_button(label="📅 Download Daily Summary (CSV)", data=results_df.to_csv(index=False), file_name='thermal_storage_daily.csv', mime='text/csv')
        else:
            st.info("🔍 Run optimization to see results and download options.")
else:
    st.info("👈 Please upload a CSV file or configure API access using the sidebar to begin.")
    with st.expander("📋 Data Source Guide"):
        st.markdown("""
        This app supports three data sources:

        ---

        ### 📊 Built-in EPEX 2024 Data
        Use the pre-loaded EPEX 2024 price dataset (`idprices-epex2024.csv`) for quick testing and analysis. This dataset contains:
        - Historical day-ahead electricity prices for 2024
        - Data in long format (automatically transformed)
        - No additional setup required

        ---

        ### 🔌 EnAppSys API
        Fetch real-time data directly from EnAppSys. You'll need:
        - Valid EnAppSys credentials (username/password)
        - Appropriate API access permissions

        **Popular Chart Codes:**
        - `gb/elec/pricing/daprices` - UK Day-Ahead Prices
        - `de/elec/pricing/daprices` - German Day-Ahead Prices  
        - `fr/elec/pricing/daprices` - French Day-Ahead Prices
        - `nl/elec/pricing/daprices` - Netherlands Day-Ahead Prices
        - `es/elec/pricing/daprices` - Spanish Day-Ahead Prices

        **Popular Bulk Data Types:**
        - `NL_SOLAR_FORECAST` - Netherlands Solar Forecast
        - `DE_WIND_FORECAST` - German Wind Forecast
        - `FR_DEMAND_FORECAST` - French Demand Forecast
        - `GB_DEMAND_FORECAST` - UK Demand Forecast

        ---

        ### 📁 File Upload
        Upload your own CSV files in the following formats:

        #### 1. Price Data Format
        **Long Format (when "Transform data" is checked):**
        - A column with datetime information (e.g., `Date (CET)`).
        - A column with the price/value (e.g., `Day Ahead Price`).
        - *Example:* `idprices-epexshort.csv`

        **Wide Format (when "Transform data" is unchecked):**
        - A 'date' column (YYYY-MM-DD) and 96 columns for each 15-minute interval (`00:00:00`, `00:15:00`, etc.).

        ---

        #### 2. Customer Demand Data Format (Optional)
        If you select "Upload Demand Profile", the file must be in **long format**. The ETL process will convert it automatically.
        - A column named `Date (CET)` with datetime information.
        - A column named `MW-th` with the thermal demand value.

        **Example (`Example_Customer Demand.csv`):**
        ```csv
        Date (CET),MW-th
        [01/01/2024 00:00],0.5
        [01/01/2024 00:15],0.5
        ...
        ```
        ---

        #### 3. Peak Period Restriction Data Format (Optional)
        If you select "Upload CSV File" for Peak Period Restrictions, the file must have a specific format:
        - It must be a CSV file in **long format** (similar to price and demand data).
        - It must contain a column named `Date (CET)` with datetime information.
        - It must contain a column named `Is HLF` with values of 1 (restricted) or 0 (not restricted).
        - The ETL process will automatically convert it to wide format for processing.

        **Example (`peak_restrictions.csv`):**
        ```csv
        Date (CET),Is HLF
        [01/01/2024 08:00],1
        [01/01/2024 08:15],1
        [01/01/2024 08:30],1
        [01/01/2024 08:45],1
        [01/01/2024 09:00],0
        ...
        ```
        """)

# Footer
st.markdown("---")