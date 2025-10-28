import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.core_imports import *
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# Load the ERA5-LAND Monthly dataset
dataset = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .select(["dewpoint_temperature_2m", "total_precipitation_sum", "temperature_2m"])
           .filterDate("1960-01-01", "2022-12-31"))

# --- Geospatial Feature Processing (Modified for Area) ---

# Load the state features for FIPS to name mapping and centroids
stateFeatures = ee.FeatureCollection("TIGER/2018/States")
non_conus_fips = ['02', '15', '60', '66', '69', '72', '78']  # AK, HI, AS, GU, MP, PR, VI
filter_non_conus = ee.Filter.inList('STATEFP', non_conus_fips).Not()
stateFeatures_conus = stateFeatures.filter(filter_non_conus)

# 1. Calculate Area for States (in square meters)
def calculate_area_state(feature):
    area_sqm = feature.area(ee.ErrorMargin(10))
    return feature.set("State_Area", area_sqm)

stateFeatures_with_area = stateFeatures_conus.map(calculate_area_state)
statePoints = stateFeatures_with_area.map(
    lambda f: f.centroid(ee.ErrorMargin(10)).copyProperties(f, ["NAME", "State_Area"])
)

# Load the county features and compute centroids
countyFeatures = ee.FeatureCollection("TIGER/2018/Counties")

# 2. Calculate Area for Counties (in square meters)
def calculate_area_county(feature):
    area_sqm = feature.area(ee.ErrorMargin(10))
    return feature.set("County_Area", area_sqm)

countyFeatures_with_area = countyFeatures.map(calculate_area_county)
countyPoints = countyFeatures_with_area.map(
    lambda f: f.centroid(ee.ErrorMargin(10)).copyProperties(f, ["NAME", "STATEFP", "County_Area"])
)

# Fetch state FIPS to name mapping and state areas (for later merging)
print("Fetching state FIPS mapping and areas...")
try:
    states_list = stateFeatures_with_area.getInfo()
    state_dict = {f['properties']['STATEFP']: f['properties']['NAME'] for f in states_list['features'] if 'STATEFP' in f['properties'] and 'NAME' in f['properties']}
    # Extract FIPS and Area for separate merge
    state_area_data = {
        f['properties']['NAME']: f['properties']['State_Area']
        for f in states_list['features'] if 'NAME' in f['properties'] and 'State_Area' in f['properties']
    }
except Exception as e:
    print(f"Error fetching state features: {e}")
    raise

# --- Functions (Precipitation Correction) ---

# NOTE: Since the original data was ERA5-LAND MONTHLY_AGGR,
# 'total_precipitation_sum' is the sum for the *month*. To get the *annual* sum,
# you must use .sum() over the year's image collection, not .mean() as was incorrectly noted
# in the previous version's comment but not the code itself.
def annualData(year):
    startDate = ee.Date.fromYMD(year, 1, 1)
    endDate = startDate.advance(1, "year")
    
    # Correct aggregation for Annual Data
    dewpoint_mean = dataset.select("dewpoint_temperature_2m").filterDate(startDate, endDate).mean()
    temp_mean = dataset.select("temperature_2m").filterDate(startDate, endDate).mean()
    precip_sum = dataset.select("total_precipitation_sum").filterDate(startDate, endDate).sum() # Use .sum() for total annual precipitation
    
    annual = dewpoint_mean.addBands(temp_mean).addBands(precip_sum).set("year", year)
    return annual

# Function to extract annual data at each state centroid
def extractState(image):
    year = image.get("year")
    pointStats = image.sampleRegions(**{
        "collection": statePoints,
        "scale": 10000,
        # Now we also request State_Area
        "properties": ["NAME", "State_Area"], 
        "tileScale": 2
    }).map(lambda feature: ee.Feature(None, {
        "State_Name": feature.get("NAME"),
        "Year": ee.Number(year),
        "dewpoint_temperature_2m": feature.get("dewpoint_temperature_2m"),
        "total_precipitation_sum": feature.get("total_precipitation_sum"),
        "temperature_2m": feature.get("temperature_2m"),
        "State_Area": feature.get("State_Area")
    }))
    return pointStats

# Function to extract annual data at each county centroid
def extractCounty(image):
    year = image.get("year")
    pointStats = image.sampleRegions(**{
        "collection": countyPoints,
        "scale": 10000,
        # Now we also request County_Area
        "properties": ["NAME", "STATEFP", "County_Area"], 
        "tileScale": 2
    }).map(lambda feature: ee.Feature(None, {
        "County_Name": feature.get("NAME"),
        "STATEFP": feature.get("STATEFP"),
        "Year": ee.Number(year),
        "dewpoint_temperature_2m": feature.get("dewpoint_temperature_2m"),
        "total_precipitation_sum": feature.get("total_precipitation_sum"),
        "temperature_2m": feature.get("temperature_2m"),
        "County_Area": feature.get("County_Area")
    }))
    return pointStats

# --- Data Fetching Loop (Modified to handle new properties) ---

years = list(range(1960, 2023))
data_state = []
data_county = []

for year in years:
    print(f"Processing year {year}...")
    annual_image = annualData(year)
    
    # Fetch state data for the year
    results_state = extractState(annual_image)
    try:
        results_state_list = results_state.getInfo()
        features_state = results_state_list['features']
        for feature in features_state:
            props = feature['properties']
            data_state.append({
                "State_Name": props.get("State_Name"),
                "Year": props.get("Year"),
                "dewpoint_temperature_2m": props.get("dewpoint_temperature_2m"),
                "total_precipitation_sum": props.get("total_precipitation_sum"),
                "temperature_2m": props.get("temperature_2m"),
                "State_Area": props.get("State_Area") # Added State_Area
            })
    except Exception as e:
        print(f"Error fetching state results for year {year}: {e}")
        raise
    
    # Fetch county data for the year
    results_county = extractCounty(annual_image)
    try:
        results_county_list = results_county.getInfo()
        features_county = results_county_list['features']
        for feature in features_county:
            props = feature['properties']
            data_county.append({
                "County_Name": props.get("County_Name"),
                "STATEFP": props.get("STATEFP"),
                "Year": props.get("Year"),
                "dewpoint_temperature_2m": props.get("dewpoint_temperature_2m"),
                "total_precipitation_sum": props.get("total_precipitation_sum"),
                "temperature_2m": props.get("temperature_2m"),
                "County_Area": props.get("County_Area") # Added County_Area
            })
    except Exception as e:
        print(f"Error fetching county results for year {year}: {e}")
        raise

# Convert to DataFrames
df_state = pd.DataFrame(data_state).drop_duplicates(subset=["State_Name", "Year"])
df_county = pd.DataFrame(data_county).drop_duplicates(subset=["County_Name", "STATEFP", "Year"])

# --- DataFrame Calculations (Modified for Area Ratio) ---

# Magnus-Tetens formula for vapor pressures
def magnus_tetens(T):
    return 6.1078 * 10 ** ((7.5 * T) / (T + 237.3))

# Function to compute RH, precip in mm, temp in C
def compute_variables(df):
    df["T_d"] = df["dewpoint_temperature_2m"] - 273.15
    df["temp"] = df["temperature_2m"] - 273.15
    df["e"] = magnus_tetens(df["T_d"])
    df["e_s"] = magnus_tetens(df["temp"])
    df["RH"] = 100 * (df["e"] / df["e_s"])
    df["RH"] = df["RH"].clip(0, 100)
    # Convert total_precipitation_sum (m) to precip (mm)
    df["precip"] = df["total_precipitation_sum"] * 1000
    return df

# Compute for state and county
df_state = compute_variables(df_state)
df_county = compute_variables(df_county)

# Add State_Name to df_county
df_county["State_Name"] = df_county["STATEFP"].map(state_dict)

# Filter to CONUS states
conus_states = set(df_state["State_Name"].unique())
df_county = df_county[df_county["State_Name"].isin(conus_states)].dropna(subset=["State_Name"])

# Merge to get state values for each county row
# We merge the area columns as well
merge_cols_state = ["State_Name", "Year", "RH", "precip", "temp", "State_Area"]
df_ratios = pd.merge(df_county, df_state[merge_cols_state],
                     on=["State_Name", "Year"], suffixes=("_county", "_state"))

# Compute ratios
df_ratios["Pr_ratio"] = df_ratios["precip_county"] / df_ratios["precip_state"]
df_ratios["Temp_ratio"] = df_ratios["temp_county"] / df_ratios["temp_state"]
df_ratios["RH_ratio"] = df_ratios["RH_county"] / df_ratios["RH_state"]
df_ratios["area_ratio"] = df_ratios["County_Area"] / df_ratios["State_Area"] # Added area_ratio

# Select final columns: Including area_ratio
final_columns = [
    "Year", 
    "State_Name", 
    "County_Name", 
    "precip_county", # County-level precipitation (mm)
    "temp_county",   # County-level temperature (°C)
    "RH_county",     # County-level relative humidity (%)
    "Pr_ratio", 
    "Temp_ratio", 
    "RH_ratio",
    "area_ratio",    # Added area_ratio
]
df_final = df_ratios[final_columns]

# --- Saving to Feather format ---

# Save to the specified directory
target_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\climatic_factors"
# Updated file name to reflect the inclusion of the area ratio
file_name = "Annual_Climate_factors_County_Data_Ratios_and_Area_1960_2022.feather"
target_path = os.path.join(target_dir, file_name)

# Ensure the target directory exists
try:
    os.makedirs(target_dir, exist_ok=True)
except Exception as e:
    print(f"Error creating directory {target_dir}: {e}")
    raise

# Save the DataFrame to Feather
df_final.to_feather(target_path) 
print(f"File successfully saved to {target_path} in FEATHER format. ✅")

# Display the first few rows
print("\nPreview of the saved DataFrame:")
print(df_final.head())