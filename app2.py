import ee
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from scipy.stats import linregress, skew, kurtosis, pearsonr
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import fiona
import zipfile
import tempfile
import os
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Climate Data Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ZipCodeStack API configuration
ZIPCODE_API_KEY = "zip_live_61Fw0aEgouDWn2ZO0sOk0aE0xnB68nWGv8mUwtsu"
ZIPCODE_BASE_URL = "https://app.zipcodestack.com/dashboard"

# Initialize Earth Engine
@st.cache_resource
def initialize_ee():
    try:
        ee.Initialize(project='ee-singhanil854')
        return True
    except Exception as e:
        st.error(f"Earth Engine initialization failed: {str(e)}")
        try:
            ee.Authenticate()
            ee.Initialize(project='ee-singhanil854')
            return True
        except:
            return False

if not initialize_ee():
    st.stop()

# Initialize session state for drawn geometry
if 'drawn_geometry' not in st.session_state:
    st.session_state.drawn_geometry = None
if 'use_drawn_geometry' not in st.session_state:
    st.session_state.use_drawn_geometry = False

# Title and description
st.title("🌍 Enhanced Climate Data Dashboard")
st.markdown("**ERA5-Land and CHIRPS Comprehensive Analysis Platform**")

# ZipCodeStack API functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_postal_codes(query, country_code=None):
    """Search postal codes using ZipCodeStack API"""
    try:
        url = f"{ZIPCODE_BASE_URL}/search"
        params = {
            'apikey': ZIPCODE_API_KEY,
            'codes': query
        }
        if country_code:
            params['country'] = country_code
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                return data['results']
        return []
    except Exception as e:
        st.error(f"Error searching postal codes: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_postal_code_details(postal_code, country_code=None):
    """Get detailed information for a specific postal code"""
    try:
        url = f"{ZIPCODE_BASE_URL}/search"
        params = {
            'apikey': ZIPCODE_API_KEY,
            'codes': postal_code
        }
        if country_code:
            params['country'] = country_code
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                result = data['results'][0]
                return {
                    'postal_code': result.get('postal_code', postal_code),
                    'city': result.get('city', 'Unknown'),
                    'state': result.get('state', 'Unknown'),
                    'country': result.get('country_code', 'Unknown'),
                    'latitude': float(result.get('latitude', 0)),
                    'longitude': float(result.get('longitude', 0))
                }
        return None
    except Exception as e:
        st.error(f"Error getting postal code details: {str(e)}")
        return None

# Sidebar configuration
st.sidebar.header("🌍 Area of Interest (AOI)")
aoi_method = st.sidebar.selectbox(
    "Select AOI Method", 
    ["Global", "India", "Postal/PIN Code", "Draw AOI", "Upload File", "Country Selection"]
)

# Initialize default geometry and map settings
geometry = ee.Geometry.Rectangle([-180, -90, 180, 90])
map_center = [20, 0]
zoom_start = 2

# Country boundaries
country_bounds = {
    "India": [68, 8, 98, 38],
    "USA": [-125, 25, -66, 49],
    "China": [73, 18, 135, 53],
    "Brazil": [-74, -34, -34, 5],
    "Australia": [113, -44, 154, -10],
    "Europe": [-10, 35, 30, 71],
    "Canada": [-141, 42, -52, 84],
    "Japan": [129, 31, 146, 46],
    "South Korea": [124, 33, 132, 39],
    "United Kingdom": [-8, 50, 2, 61]
}

# Country codes for postal code search
country_codes = {
    "United States": "US",
    "India": "IN",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "Germany": "DE",
    "France": "FR",
    "Japan": "JP",
    "South Korea": "KR",
    "Brazil": "BR",
    "China": "CN",
    "Russia": "RU",
    "Italy": "IT",
    "Spain": "ES",
    "Netherlands": "NL",
    "Sweden": "SE",
    "Norway": "NO",
    "Denmark": "DK",
    "Finland": "FI",
    "Switzerland": "CH"
}

# AOI Selection Logic
def create_buffer_geometry(coords, buffer_km=50):
    """Create a buffered geometry around coordinates"""
    point = ee.Geometry.Point(coords)
    return point.buffer(buffer_km * 1000)  # Convert km to meters

if aoi_method == "India":
    geometry = ee.Geometry.Rectangle(country_bounds["India"])
    map_center = [20, 78]
    zoom_start = 5
    st.session_state.use_drawn_geometry = False

elif aoi_method == "Postal/PIN Code":
    st.sidebar.subheader("🏘️ Postal/PIN Code Search")
    
    # Country selection for postal code search
    selected_country = st.sidebar.selectbox(
        "Select Country (Optional)",
        ["Any"] + list(country_codes.keys())
    )
    
    country_code = None if selected_country == "Any" else country_codes.get(selected_country)
    
    # Postal code input
    postal_code_input = st.sidebar.text_input(
        "Enter Postal/PIN Code",
        placeholder="e.g., 110001, 10001, SW1A 1AA"
    )
    
    if postal_code_input:
        with st.spinner("🔍 Searching postal code..."):
            postal_info = get_postal_code_details(postal_code_input, country_code)
            
            if postal_info:
                with st.sidebar:
                    st.success(f"✅ Found: {postal_info['city']}, {postal_info['state']}, {postal_info['country']}")
                    buffer_size = st.slider("Buffer Size (km)", 10, 200, 50)
                
                coords = [postal_info['longitude'], postal_info['latitude']]
                geometry = create_buffer_geometry(coords, buffer_size)
                map_center = [postal_info['latitude'], postal_info['longitude']]
                zoom_start = 10
                st.session_state.use_drawn_geometry = False
            else:
                st.sidebar.error("❌ Postal code not found. Please check the code and try again.")


elif aoi_method == "Country Selection":
    country = st.sidebar.selectbox("Select Country/Region", list(country_bounds.keys()))
    bounds = country_bounds[country]
    geometry = ee.Geometry.Rectangle(bounds)
    map_center = [(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2]
    zoom_start = 4
    st.session_state.use_drawn_geometry = False

elif aoi_method == "Draw AOI":
    st.sidebar.info("🖊️ Use the drawing tools on the map below")
    st.sidebar.markdown("""
    **Instructions:**
    1. Use the polygon drawing tool on the map
    2. Draw your area of interest
    3. Click 'Apply Drawn AOI' button
    4. Run the analysis
    """)
    
    # Check if we should use drawn geometry
    if st.session_state.use_drawn_geometry and st.session_state.drawn_geometry:
        try:
            geometry = ee.Geometry(st.session_state.drawn_geometry)
            # Calculate center of drawn geometry
            centroid = geometry.centroid().coordinates().getInfo()
            map_center = [centroid[1], centroid[0]]
            zoom_start = 8
            st.sidebar.success("✅ Using drawn AOI")
        except:
            st.sidebar.warning("⚠️ Error with drawn geometry, using global extent")

elif aoi_method == "Upload File":
    st.sidebar.markdown("📁 **Upload KML/KMZ/Shapefile**")
    uploaded_file = st.sidebar.file_uploader(
        "Choose file", 
        type=["kml", "kmz", "zip", "shp"],
        help="Support: KML, KMZ, or ZIP containing shapefile"
    )
    
    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith(".kmz"):
                    with zipfile.ZipFile(file_path, 'r') as kmz:
                        kmz.extractall(tmpdir)
                        kml_files = [f for f in kmz.namelist() if f.endswith(".kml")]
                        if kml_files:
                            file_path = os.path.join(tmpdir, kml_files[0])
                
                elif uploaded_file.name.endswith(".zip"):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmpdir)
                        shp_files = [f for f in zip_ref.namelist() if f.endswith(".shp")]
                        if shp_files:
                            file_path = os.path.join(tmpdir, shp_files[0])
                
                gdf = gpd.read_file(file_path)
                if not gdf.empty and gdf.geometry.iloc[0] is not None:
                    geom = gdf.geometry.iloc[0]
                    if geom.geom_type == "Polygon":
                        coords = [list(geom.exterior.coords)]
                        geometry = ee.Geometry.Polygon(coords)
                    elif geom.geom_type == "MultiPolygon":
                        coords = [list(poly.exterior.coords) for poly in geom.geoms]
                        geometry = ee.Geometry.MultiPolygon(coords)
                    
                    centroid = geometry.centroid().coordinates().getInfo()
                    map_center = [centroid[1], centroid[0]]
                    zoom_start = 8
                    st.sidebar.success("✅ File uploaded successfully!")
                    st.session_state.use_drawn_geometry = False
                
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")

# Date selection with presets
st.sidebar.header("📅 Date Range Selection")

# Date presets
date_presets = {
    "Last Year": (datetime.now() - timedelta(days=365), datetime.now() - timedelta(days=1)),
    "Last 6 Months": (datetime.now() - timedelta(days=180), datetime.now() - timedelta(days=1)),
    "Last 3 Months": (datetime.now() - timedelta(days=90), datetime.now() - timedelta(days=1)),
    "2023": (datetime(2023, 1, 1), datetime(2023, 12, 31)),
    "2022": (datetime(2022, 1, 1), datetime(2022, 12, 31)),
    "2021": (datetime(2021, 1, 1), datetime(2021, 12, 31)),
    "Custom": None
}

preset = st.sidebar.selectbox("Quick Date Selection", list(date_presets.keys()))

if preset == "Custom" or date_presets[preset] is None:
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=datetime(2023, 1, 1),
        min_value=datetime(1981, 1, 1),
        max_value=datetime.now() - timedelta(days=1)
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        value=datetime(2023, 12, 31),
        min_value=start_date,
        max_value=datetime.now() - timedelta(days=1)
    )
else:
    start_date, end_date = date_presets[preset]
    st.sidebar.info(f"📅 Selected: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Analysis options
st.sidebar.header("🔧 Analysis Options")
analysis_type = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Time Series", "Statistics", "Trend Analysis", "Correlation Analysis", "Anomaly Detection", "Seasonal Analysis"],
    default=["Time Series", "Statistics"]
)

temporal_aggregation = st.sidebar.selectbox(
    "Temporal Aggregation",
    ["Daily", "Weekly", "Monthly", "Seasonal", "Annual"]
)

# Convert dates to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Data loading function
def load_datasets(start_date_str, end_date_str):
    """Load datasets"""
    try:
        # ERA5-Land dataset
        era5_land = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').filterDate(start_date_str, end_date_str)
        
        # CHIRPS dataset
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start_date_str, end_date_str)
        
        return era5_land, chirps
    except Exception as e:
        st.error(f"Failed to load datasets: {str(e)}")
        return None, None

# Load datasets
with st.spinner("🔄 Loading datasets..."):
    era5_land, chirps = load_datasets(start_date_str, end_date_str)

if era5_land is None or chirps is None:
    st.error("Failed to load datasets. Please check your connection and try again.")
    st.stop()

# Available bands
era5_bands = {
    'temperature_2m_max': 'Maximum Temperature (2m)',
    'temperature_2m_min': 'Minimum Temperature (2m)',
    'temperature_2m': 'Temperature (2m)',
    'dewpoint_temperature_2m': 'Dewpoint Temperature (2m)',
    'total_precipitation_sum': 'Total Precipitation',
    'surface_pressure': 'Surface Pressure',
    'u_component_of_wind_10m': 'Wind U-component (10m)',
    'v_component_of_wind_10m': 'Wind V-component (10m)',
    'surface_solar_radiation_downwards_sum': 'Solar Radiation',
    'surface_thermal_radiation_downwards_sum': 'Thermal Radiation',
    'total_evaporation_sum': 'Total Evaporation',
    'potential_evaporation_sum': 'Potential Evaporation',
    'runoff_sum': 'Runoff',
    'sub_surface_runoff_sum': 'Sub-surface Runoff'
}

chirps_bands = {
    'precipitation': 'CHIRPS Precipitation'
}

# Band selection
selected_era5_bands = st.sidebar.multiselect(
    "Select ERA5-Land Variables",
    list(era5_bands.keys()),
    default=['temperature_2m_max', 'temperature_2m_min', 'total_precipitation_sum'],
    format_func=lambda x: era5_bands[x]
)

include_chirps = st.sidebar.checkbox("Include CHIRPS Precipitation", value=True)

# Visualization parameters
vis_params = {
    'temperature_2m_max': {'min': 250, 'max': 320, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff', '00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00', 'ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']},
    'temperature_2m_min': {'min': 230, 'max': 300, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff', '00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00', 'ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']},
    'temperature_2m': {'min': 240, 'max': 310, 'palette': ['000080', '0000d9', '4000ff', '8000ff', '0080ff', '00ffff', '00ff80', '80ff00', 'daff00', 'ffff00', 'fff500', 'ffda00', 'ffb000', 'ffa400', 'ff4f00', 'ff2500', 'ff0a00', 'ff00ff']},
    'total_precipitation_sum': {'min': 0, 'max': 0.1, 'palette': ['ffffff', '00ffff', '0080ff', 'da00ff', 'ffa400', 'ff0000']},
    'precipitation': {'min': 0, 'max': 50, 'palette': ['ffffff', '00ffff', '0080ff', 'da00ff', 'ffa400', 'ff0000']},
    'surface_pressure': {'min': 95000, 'max': 105000, 'palette': ['blue', 'green', 'yellow', 'red']},
    'dewpoint_temperature_2m': {'min': 230, 'max': 300, 'palette': ['purple', 'blue', 'green', 'yellow', 'red']},
    'u_component_of_wind_10m': {'min': -20, 'max': 20, 'palette': ['red', 'white', 'blue']},
    'v_component_of_wind_10m': {'min': -20, 'max': 20, 'palette': ['red', 'white', 'blue']},
    'surface_solar_radiation_downwards_sum': {'min': 0, 'max': 30000000, 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']},
    'total_evaporation_sum': {'min': -0.01, 'max': 0.01, 'palette': ['brown', 'yellow', 'green', 'blue']},
    'runoff_sum': {'min': 0, 'max': 0.001, 'palette': ['white', 'blue', 'green', 'red']}
}

# Data extraction function
def extract_time_series_data(collection, band, geometry_json, start_date, end_date):
    """Extract time series data from image collection"""
    try:
        geometry = ee.Geometry(geometry_json)
        
        def extract_values(image):
            stats = image.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9
            )
            return ee.Feature(None, {
                'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                band: stats.get(band)
            })
        
        features = collection.map(extract_values)
        data = features.getInfo()
        
        # Convert to DataFrame
        rows = []
        for feature in data['features']:
            props = feature['properties']
            if props[band] is not None:
                rows.append({
                    'date': pd.to_datetime(props['date']),
                    'value': props[band]
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.set_index('date', inplace=True)
            return df.sort_index()
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error extracting data for {band}: {str(e)}")
        return pd.DataFrame()

# Convert geometry to JSON for processing
geometry_json = geometry.getInfo()

# Create map for AOI visualization and drawing
st.header("🗺️ Area of Interest")

# Create map
m = folium.Map(location=map_center, zoom_start=zoom_start)

# Add drawing tools for Draw AOI method
if aoi_method == "Draw AOI":
    from folium.plugins import Draw
    
    draw = Draw(
        export=True,
        filename='data.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

# Add current AOI boundary if not drawing
if aoi_method != "Draw AOI" or st.session_state.use_drawn_geometry:
    try:
        if geometry_json['type'] == 'Polygon':
            coords = geometry_json['coordinates'][0]
            folium.Polygon(
                locations=[[lat, lon] for lon, lat in coords],
                color='red',
                weight=2,
                fill=False,
                popup='Area of Interest'
            ).add_to(m)
        elif geometry_json['type'] == 'Rectangle':
            bounds = geometry_json['coordinates'][0]
            folium.Rectangle(
                bounds=[[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]],
                color='red',
                weight=2,
                fill=False,
                popup='Area of Interest'
            ).add_to(m)
    except Exception as e:
        st.warning(f"Could not display AOI boundary: {str(e)}")

# Display map
map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked_popup", "all_drawings"])

# Handle drawn AOI
if aoi_method == "Draw AOI":
    if map_data.get('all_drawings') and len(map_data['all_drawings']) > 0:
        st.info("🖊️ AOI drawn on map. Click 'Apply Drawn AOI' to use it.")
        
        if st.button("✅ Apply Drawn AOI", type="primary"):
            try:
                latest_drawing = map_data['all_drawings'][-1]  # Get the most recent drawing
                drawn_geom = latest_drawing['geometry']
                
                if drawn_geom['type'] == 'Polygon':
                    coords = drawn_geom['coordinates']
                    st.session_state.drawn_geometry = {
                        'type': 'Polygon',
                        'coordinates': coords
                    }
                    st.session_state.use_drawn_geometry = True
                    st.success("✅ AOI updated! You can now run the analysis.")
                    st.rerun()
                else:
                    st.error("❌ Please draw a polygon area")
                    
            except Exception as e:
                st.error(f"❌ Error processing drawn AOI: {str(e)}")
    else:
        st.info("ℹ️ Draw a polygon on the map to define your area of interest")

# Main analysis
if st.button("🚀 Run Analysis", type="primary", key="main_analysis"):
    with st.spinner("🔄 Processing data..."):
        # Use drawn geometry if available and selected
        if st.session_state.use_drawn_geometry and st.session_state.drawn_geometry:
            geometry_json = st.session_state.drawn_geometry
        
        # Extract data for selected bands
        data_dict = {}
        
        # ERA5 data
        for band in selected_era5_bands:
            df = extract_time_series_data(era5_land, band, geometry_json, start_date_str, end_date_str)
            if not df.empty:
                data_dict[band] = df
        
        # CHIRPS data
        if include_chirps:
            df = extract_time_series_data(chirps, 'precipitation', geometry_json, start_date_str, end_date_str)
            if not df.empty:
                data_dict['chirps_precipitation'] = df
    
    if not data_dict:
        st.error("❌ No data extracted. Please check your AOI and date range.")
        st.stop()
    
    # Store data in session state
    st.session_state['data_dict'] = data_dict
    st.session_state['analysis_complete'] = True

# Analysis results (keeping the same structure as before)
if st.session_state.get('analysis_complete', False) and 'data_dict' in st.session_state:
    data_dict = st.session_state['data_dict']
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Time Series", "📈 Statistics", "🔍 Trend Analysis", "🔗 Correlations", "🗺️ Data Maps"])
    
    with tab1:
        st.header("📊 Time Series Analysis")
        
        # Temperature analysis
        temp_data = {}
        for band in ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m']:
            if band in data_dict:
                temp_data[band] = data_dict[band]['value'] - 273.15  # Convert to Celsius
        
        if temp_data:
            fig = go.Figure()
            colors = ['red', 'blue', 'green']
            for i, (band, data) in enumerate(temp_data.items()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=era5_bands[band],
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title="Temperature Time Series",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation comparison
        precip_data = {}
        if 'total_precipitation_sum' in data_dict:
            precip_data['ERA5'] = data_dict['total_precipitation_sum']['value'] * 1000  # Convert to mm
        if 'chirps_precipitation' in data_dict:
            precip_data['CHIRPS'] = data_dict['chirps_precipitation']['value']
        
        if precip_data:
            fig = go.Figure()
            colors = ['blue', 'orange']
            for i, (source, data) in enumerate(precip_data.items()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines',
                    name=f"{source} Precipitation",
                    line=dict(color=colors[i])
                ))
            
            fig.update_layout(
                title="Precipitation Comparison",
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📈 Statistical Summary")
        
        # Calculate comprehensive statistics
        stats_data = []
        for band, df in data_dict.items():
            values = df['value'].values
            
            # Unit conversion
            if 'temperature' in band:
                values = values - 273.15  # K to °C
                unit = "°C"
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000  # m to mm
                unit = "mm"
            elif band == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Calculate statistics
            stats = {
                'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                'Count': len(values),
                'Mean': f"{np.mean(values):.2f} {unit}",
                'Std Dev': f"{np.std(values):.2f} {unit}",
                'Min': f"{np.min(values):.2f} {unit}",
                'Max': f"{np.max(values):.2f} {unit}",
                'Range': f"{np.max(values) - np.min(values):.2f} {unit}",
                'Skewness': f"{skew(values):.2f}",
                'Kurtosis': f"{kurtosis(values):.2f}",
                'CV': f"{np.std(values)/np.mean(values)*100:.1f}%"
            }
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Distribution plots
        st.subheader("📊 Data Distributions")
        
        # Create subplot for distributions
        n_vars = len(data_dict)
        cols = min(3, n_vars)
        rows = (n_vars - 1) // cols + 1
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[era5_bands.get(band, band) for band in data_dict.keys()]
        )
        
        
        
        for i, (band, df) in enumerate(data_dict.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            values = df['value'].values
            if 'temperature' in band:
                values = values - 273.15
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000
            
            fig.add_trace(
                go.Histogram(x=values, name=band, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=300*rows, title_text="Variable Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 Trend Analysis")
        
        # Calculate trends for each variable
        trend_data = []
        for band, df in data_dict.items():
            values = df['value'].values
            dates = pd.to_datetime(df.index)
            time_numeric = (dates - dates[0]).days
            
            # Unit conversion
            if 'temperature' in band:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in band and band != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif band == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Calculate trend
            slope, intercept, r_value, p_value, std_err = linregress(time_numeric, values)
            
            # Convert slope to per year
            slope_per_year = slope * 365.25
            
            trend_data.append({
                'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                'Trend (per year)': f"{slope_per_year:.4f} {unit}/year",
                'R-squared': f"{r_value**2:.3f}",
                'P-value': f"{p_value:.4f}",
                'Significance': "Yes" if p_value < 0.05 else "No"
            })
        
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True)
        
        # Trend visualization
        st.subheader("📈 Trend Visualization")
        
        selected_var = st.selectbox(
            "Select variable for trend plot",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="trend_var_select"
        )
        
        if selected_var:
            df = data_dict[selected_var]
            values = df['value'].values
            dates = df.index
            
            # Unit conversion
            if 'temperature' in selected_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in selected_var and selected_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif selected_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Calculate trend line
            time_numeric = (pd.to_datetime(dates) - pd.to_datetime(dates[0])).days
            slope, intercept, _, _, _ = linregress(time_numeric, values)
            trend_line = slope * time_numeric + intercept
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=values,
                mode='lines',
                name='Data',
                line=dict(color='blue', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Trend Analysis: {era5_bands.get(selected_var, selected_var)}",
                xaxis_title="Date",
                yaxis_title=f"Value ({unit})",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔗 Correlation Analysis")
        
        if len(data_dict) >= 2:
            # Create correlation matrix
            correlation_data = {}
            for band, df in data_dict.items():
                values = df['value'].values
                if 'temperature' in band:
                    values = values - 273.15
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                correlation_data[band] = values
            
            # Align data by dates
            common_dates = None
            for band, df in data_dict.items():
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))
            
            if common_dates:
                aligned_data = {}
                for band in correlation_data:
                    df = data_dict[band]
                    aligned_data[band] = df.loc[list(common_dates)]['value'].values
                    if 'temperature' in band:
                        aligned_data[band] = aligned_data[band] - 273.15
                    elif 'precipitation' in band and band != 'chirps_precipitation':
                        aligned_data[band] = aligned_data[band] * 1000
                
                corr_df = pd.DataFrame(aligned_data)
                correlation_matrix = corr_df.corr()
                
                # Heatmap
                fig = px.imshow(
                    correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=[era5_bands.get(col, col) for col in correlation_matrix.columns],
                    y=[era5_bands.get(row, row) for row in correlation_matrix.index],
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig.update_layout(title="Correlation Matrix", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation table
                st.subheader("Correlation Coefficients")
                st.dataframe(correlation_matrix, use_container_width=True)
        else:
            st.info("Need at least 2 variables for correlation analysis")
    
    with tab5:
        st.header("🗺️ Spatial Visualization")
        
        # Create map
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Add data layers
        for band in selected_era5_bands:
            try:
                image = era5_land.select(band).mean().clip(ee.Geometry(geometry_json))
                vis_param = vis_params.get(band, {'min': 0, 'max': 1, 'palette': ['blue', 'green', 'yellow', 'red']})
                
                map_id = image.getMapId(vis_param)
                folium.raster_layers.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name=era5_bands[band],
                    overlay=True,
                    control=True
                ).add_to(m)
            except Exception as e:
                st.error(f"Failed to add {band} layer: {str(e)}")
        
# Add CHIRPS layer
        if include_chirps:
            try:
                chirps_image = chirps.select('precipitation').mean().clip(ee.Geometry(geometry_json))
                map_id = chirps_image.getMapId(vis_params['precipitation'])
                folium.raster_layers.TileLayer(
                    tiles=map_id['tile_fetcher'].url_format,
                    attr='Google Earth Engine',
                    name='CHIRPS Precipitation',
                    overlay=True,
                    control=True
                ).add_to(m)
            except Exception as e:
                st.error(f"Failed to add CHIRPS layer: {str(e)}")
        
        # Add AOI boundary
        try:
            if geometry_json['type'] == 'Polygon':
                coords = geometry_json['coordinates'][0]
                folium.Polygon(
                    locations=[[lat, lon] for lon, lat in coords],
                    color='red',
                    weight=2,
                    fill=False,
                    popup='Area of Interest'
                ).add_to(m)
            elif geometry_json['type'] == 'Rectangle':
                # Handle rectangle geometry
                bounds = geometry_json['coordinates'][0]
                folium.Rectangle(
                    bounds=[[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]],
                    color='red',
                    weight=2,
                    fill=False,
                    popup='Area of Interest'
                ).add_to(m)
        except Exception as e:
            st.warning(f"Could not add AOI boundary: {str(e)}")
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Handle drawn AOI
        if aoi_method == "Draw AOI":
            if map_data['last_object_clicked_popup'] or map_data['all_drawings']:
                if st.button("Apply Drawn AOI"):
                    try:
                        if map_data['all_drawings']:
                            drawn_geom = map_data['all_drawings'][0]['geometry']
                            if drawn_geom['type'] == 'Polygon':
                                coords = drawn_geom['coordinates']
                                geometry = ee.Geometry.Polygon(coords)
                                st.session_state['custom_geometry'] = geometry
                                st.success("✅ AOI updated! Click 'Run Analysis' to apply.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error processing drawn AOI: {str(e)}")

# Advanced Analysis Section
if st.session_state.get('analysis_complete', False):
    st.header("🔬 Advanced Analysis")
    
    # Anomaly Detection
    if "Anomaly Detection" in analysis_type:
        st.subheader("🚨 Anomaly Detection")
        
        anomaly_var = st.selectbox(
            "Select variable for anomaly detection",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="anomaly_var_select"
        )
        
        if anomaly_var:
            df = data_dict[anomaly_var].copy()
            values = df['value'].values
            
            # Unit conversion
            if 'temperature' in anomaly_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in anomaly_var and anomaly_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif anomaly_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Calculate anomalies (Z-score method)
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = np.abs((values - mean_val) / std_val)
            
            threshold = st.slider("Anomaly Threshold (Z-score)", 1.0, 4.0, 2.0, 0.1)
            anomalies = z_scores > threshold
            
            # Plot anomalies
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=values,
                mode='lines',
                name='Data',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df.index[anomalies],
                y=values[anomalies],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8)
            ))
            
            fig.update_layout(
                title=f"Anomaly Detection: {era5_bands.get(anomaly_var, anomaly_var)}",
                xaxis_title="Date",
                yaxis_title=f"Value ({unit})",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly statistics
            n_anomalies = np.sum(anomalies)
            st.info(f"🔍 Found {n_anomalies} anomalies ({n_anomalies/len(values)*100:.1f}% of data)")
    
    # Seasonal Analysis
    if "Seasonal Analysis" in analysis_type:
        st.subheader("🌱 Seasonal Analysis")
        
        seasonal_var = st.selectbox(
            "Select variable for seasonal analysis",
            list(data_dict.keys()),
            format_func=lambda x: era5_bands.get(x, x.replace('_', ' ').title()),
            key="seasonal_var_select"
        )
        
        if seasonal_var:
            df = data_dict[seasonal_var].copy()
            values = df['value'].values
            
            # Unit conversion
            if 'temperature' in seasonal_var:
                values = values - 273.15
                unit = "°C"
            elif 'precipitation' in seasonal_var and seasonal_var != 'chirps_precipitation':
                values = values * 1000
                unit = "mm"
            elif seasonal_var == 'chirps_precipitation':
                unit = "mm"
            else:
                unit = "units"
            
            # Add seasonal information
            df['value_converted'] = values
            df['month'] = df.index.month
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Monthly box plot
            fig1 = px.box(
                df.reset_index(),
                x='month',
                y='value_converted',
                title=f"Monthly Distribution: {era5_bands.get(seasonal_var, seasonal_var)}"
            )
            fig1.update_layout(
                xaxis_title="Month",
                yaxis_title=f"Value ({unit})",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Seasonal statistics
            seasonal_stats = df.groupby('season')['value_converted'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.subheader("📊 Seasonal Statistics")
            st.dataframe(seasonal_stats, use_container_width=True)

# Data Export Section
st.header("💾 Data Export")

if st.session_state.get('analysis_complete', False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Statistics", type="secondary"):
            # Create comprehensive statistics export
            export_data = []
            for band, df in data_dict.items():
                values = df['value'].values
                if 'temperature' in band:
                    values = values - 273.15
                    unit = "°C"
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                    unit = "mm"
                elif band == 'chirps_precipitation':
                    unit = "mm"
                else:
                    unit = "units"
                
                stats = {
                    'Variable': era5_bands.get(band, band.replace('_', ' ').title()),
                    'Unit': unit,
                    'Count': len(values),
                    'Mean': np.mean(values),
                    'Std_Dev': np.std(values),
                    'Min': np.min(values),
                    'Max': np.max(values),
                    'Range': np.max(values) - np.min(values),
                    'Skewness': skew(values),
                    'Kurtosis': kurtosis(values),
                    'CV_Percent': np.std(values)/np.mean(values)*100
                }
                export_data.append(stats)
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics CSV",
                data=csv,
                file_name=f"climate_statistics_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Time Series", type="secondary"):
            # Combine all time series data
            combined_df = pd.DataFrame()
            for band, df in data_dict.items():
                values = df['value'].values
                if 'temperature' in band:
                    values = values - 273.15
                elif 'precipitation' in band and band != 'chirps_precipitation':
                    values = values * 1000
                
                if combined_df.empty:
                    combined_df = pd.DataFrame(index=df.index)
                
                combined_df[era5_bands.get(band, band)] = values
            
            csv = combined_df.to_csv()
            st.download_button(
                label="📥 Download Time Series CSV",
                data=csv,
                file_name=f"climate_timeseries_{start_date_str}_{end_date_str}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🔗 Export Correlations", type="secondary"):
            if len(data_dict) >= 2:
                # Recreate correlation matrix for export
                correlation_data = {}
                for band, df in data_dict.items():
                    values = df['value'].values
                    if 'temperature' in band:
                        values = values - 273.15
                    elif 'precipitation' in band and band != 'chirps_precipitation':
                        values = values * 1000
                    correlation_data[era5_bands.get(band, band)] = values
                
                corr_df = pd.DataFrame(correlation_data)
                correlation_matrix = corr_df.corr()
                
                csv = correlation_matrix.to_csv()
                st.download_button(
                    label="📥 Download Correlations CSV",
                    data=csv,
                    file_name=f"climate_correlations_{start_date_str}_{end_date_str}.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
**🌍 Enhanced Climate Data Dashboard (Prepared By : Dr. Anil Kumar Singh)**  
*Powered by Google Earth Engine, ERA5-Land, and CHIRPS datasets*  
*Built with Streamlit and Plotly*
""")

# Session state cleanup option
if st.sidebar.button("🔄 Reset Analysis"):
    for key in ['data_dict', 'analysis_complete', 'custom_geometry']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()