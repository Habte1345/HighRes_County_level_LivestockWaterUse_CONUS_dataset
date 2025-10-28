import os
import sys
import random
import numpy as np
import pandas as pd

# --- Suppress TensorFlow messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all messages, 3=errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN optimizations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.core_imports import *  # if this imports TF, messages are now suppressed

# Set random seed
random.seed(42)

# Define data directory
data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"

# --- Load USGS county data for different years ---
def load_usgs_county(year, cols, rename_cols=None):
    path = os.path.join(data_dir, "usgs", "usgs_water_data_feather", f"usgs_{year}_county.feather")
    df = pd.read_feather(path)[cols]
    if rename_cols:
        df.columns = rename_cols
    return df

# 1985 and 1990
usgs_1985_county_cons = load_usgs_county(1985, ["state","lv-total","lv-cuse"], ["STATE","LV_TOTAL","LV_CU"])
usgs_1990_county_cons = load_usgs_county(1990, ["state","lv-total","lv-cuse"], ["STATE","LV_TOTAL","LV_CU"])

# 1995
usgs_1995_county_cons = load_usgs_county(1995, ["State","CountyName","LV-CUTot","LS-WTotl"], ["STATE","COUNTY_NAME","LV_TOTAL","LS_WTOTL"])

# 2005
usgs_2005_county_cons = load_usgs_county(2005, ["STATE","State-County Name","LS-WSWFr","LS-WFrTo"], ["STATE","COUNTY_NAME","LS_WSWFR","LS_WFRTO"])

# 2010
usgs_2010_county_cons = pd.read_feather(os.path.join(data_dir, "usgs", "usgs_water_data_feather", "usgs_2010_county.feather"))
usgs_2010_county_cons['LI_CU'] = usgs_2010_county_cons["LI-WFrTo"] * np.random.uniform(0.9, 0.98)
usgs_2010_county_cons = usgs_2010_county_cons[["STATE", "COUNTY", "LI-WFrTo", "LI_CU"]]

# 2015
usgs_2015_county_cons = pd.read_feather(os.path.join(data_dir, "usgs", "usgs_water_data_feather", "usgs_2015_county.feather"))
usgs_2015_county_cons['LI_CU'] = usgs_2015_county_cons["LI-WFrTo"] * np.random.uniform(0.9, 0.98)
usgs_2015_county_cons = usgs_2015_county_cons[["STATE", "COUNTY", "LI-WFrTo", "LI_CU"]]

# Align COUNTY names for 1985 and 1990 with 2010
usgs_1985_county_cons['COUNTY'] = usgs_2010_county_cons['COUNTY']
usgs_1990_county_cons['COUNTY'] = usgs_2010_county_cons['COUNTY']

# --- Compute consumption ratios ---
usgs_1985_county_cons['cons_ratio'] = usgs_1985_county_cons['LV_CU'] / usgs_1985_county_cons['LV_TOTAL']
usgs_1990_county_cons['cons_ratio'] = usgs_1990_county_cons['LV_CU'] / usgs_1990_county_cons['LV_TOTAL']
usgs_1995_county_cons['cons_ratio'] = usgs_1995_county_cons['LS_WTOTL'] / usgs_1995_county_cons['LV_TOTAL']
usgs_2005_county_cons['cons_ratio'] = usgs_2005_county_cons['LS_WSWFR'] / usgs_2005_county_cons['LS_WFRTO']
usgs_2010_county_cons['cons_ratio'] = usgs_2010_county_cons['LI_CU'] / usgs_2010_county_cons['LI-WFrTo']
usgs_2015_county_cons['cons_ratio'] = usgs_2015_county_cons['LI_CU'] / usgs_2015_county_cons['LI-WFrTo']

# --- Combine into a single DataFrame ---
usgs_county_cons_1985_2015 = pd.DataFrame({
    'COUNTY_NAME': usgs_1985_county_cons['COUNTY'],
    'STATE': usgs_1985_county_cons['STATE'],
    'cons_1985_ratio': usgs_1985_county_cons['cons_ratio'],
    'cons_1990_ratio': usgs_1990_county_cons['cons_ratio'],
    'cons_1995_ratio': usgs_1995_county_cons['cons_ratio'],
    'cons_2005_ratio': usgs_2005_county_cons['cons_ratio'],
    'cons_2010_ratio': usgs_2010_county_cons['cons_ratio'],
    'cons_2015_ratio': usgs_2015_county_cons['cons_ratio']
}).dropna().round(3)

# --- Melt the DataFrame to long format ---
usgs_melted = usgs_county_cons_1985_2015.melt(
    id_vars=['STATE', 'COUNTY_NAME'],
    value_vars=['cons_1985_ratio','cons_1990_ratio','cons_1995_ratio',
                'cons_2005_ratio','cons_2010_ratio','cons_2015_ratio'],
    var_name='Year',
    value_name='CL_cons_ratio'
)

# Extract year and set as index
usgs_melted['Year'] = usgs_melted['Year'].str.extract('(\d{4})').astype(int)
usgs_county_cons_1985_2015_final = usgs_melted.set_index('Year')

# --- Save as feather file ---
output_path = os.path.join(data_dir, "usgs", "usgs_water_data_feather", "usgs_county_cons_1985_2015.feather")
usgs_county_cons_1985_2015_final.reset_index().to_feather(output_path)

# --- Clear confirmation print ---
print(f"✅ Feather file successfully saved at:\n{output_path}")



# ------------------------------ ML data preparation for 1960_1980 ---------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from scripts.core_imports import *
except ImportError:
    print("Warning: 'core_imports' not found. Proceeding without it.")
    pass


random.seed(42)

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
ml_data_dir = os.path.join(data_dir, "ML")
os.makedirs(ml_data_dir, exist_ok=True) # Ensure the ML output directory exists

# --- State FIPS to Name Mapping ---
state_map = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
    28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada',
    33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 36: 'New York',
    37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
    42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
    53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
}

# ----------------------------------------------------------------------
# --- 1. Load and Process Annual Climate Factors (1960-1980 Subset) ---
# ----------------------------------------------------------------------
print("1. Loading and filtering Annual Climate Factors...")
climate_path = os.path.join(data_dir, "climatic_factors", "Annual_Climate_factors_County_Data_Ratios_and_Area_1960_2022.feather")
Annual_Climate_factors_County_Data_and_Ratios_1960_2022 = (
    pd.read_feather(climate_path)
    .sort_values(by=["State_Name", "County_Name", "Year"])
    .reset_index(drop=True)
)

# Prepare climate data for 1960-1980 merge
df_climate = Annual_Climate_factors_County_Data_and_Ratios_1960_2022.copy()
df_climate['Year'] = df_climate['Year'].astype(str)
Annual_Climate_factors_County_Data_and_Ratios_1960_1980 = df_climate[
    (df_climate['Year'] >= '1960') & (df_climate['Year'] <= '1980')
].copy()

# Rename State column and convert Year to integer for merging
Annual_Climate_factors_County_Data_and_Ratios_1960_1980.rename(
    columns={'State_Name': 'STATE'}, inplace=True
)
Annual_Climate_factors_County_Data_and_Ratios_1960_1980['Year'] = (
    Annual_Climate_factors_County_Data_and_Ratios_1960_1980['Year'].astype(int)
)


# ----------------------------------------------------------------------
# --- 2. Load and Process USGS SL cons_ratio (1960-1980 - State Level) ---
# ----------------------------------------------------------------------
print("2. Processing USGS State-Level Consumption Ratio (1960-1980)...")
cols = ['STATE', '1960', '1965', '1970', '1975', '1980']
usgs_1950_1980_state = os.path.join(data_dir, "usgs", "usgs_water_data_feather", "usgs_1950_1980_state.feather")
usgs_1950_1980_state_cons_ratio_raw = pd.read_feather(usgs_1950_1980_state)[cols]

# Melt the DataFrame to long format (Year and SL_cons_ratio columns)
df_melted = usgs_1950_1980_state_cons_ratio_raw.melt(
    id_vars='STATE',
    var_name='Year',
    value_name='SL_cons_ratio'
)

usgs_1950_1980_state_cons_ratio = df_melted.copy()

# Apply State FIPS mapping and drop missing (unmapped) states
usgs_1950_1980_state_cons_ratio['STATE'] = (
    usgs_1950_1980_state_cons_ratio['STATE'].map(state_map)
)
usgs_1950_1980_state_cons_ratio.dropna(inplace=True)
usgs_1950_1980_state_cons_ratio.reset_index(drop=True, inplace=True)

# Convert 'Year' to integer for merging
usgs_1950_1980_state_cons_ratio['Year'] = (
    usgs_1950_1980_state_cons_ratio['Year'].astype(int)
)


# ----------------------------------------------------------------------
# --- 3. Merge DataFrames and Finalize ---
# ----------------------------------------------------------------------
print("3. Merging and finalizing data...")
ML_data_prepared_all_1960_1980 = Annual_Climate_factors_County_Data_and_Ratios_1960_1980.merge(
    usgs_1950_1980_state_cons_ratio[['Year', 'STATE', 'SL_cons_ratio']],
    on=['Year', 'STATE'],
    how='left'
)

# Fill NaN SL_cons_ratio with 0.0, as requested
ML_data_prepared_all_1960_1980['SL_cons_ratio'] = (
    ML_data_prepared_all_1960_1980['SL_cons_ratio'].fillna(0.0)
)

# ----------------------------------------------------------------------
# --- 4. Save the Final DataFrame ---
# ----------------------------------------------------------------------
output_filename = 'ML_data_prepared_all_1960_1980.feather'
output_path = os.path.join(ml_data_dir, output_filename)
ML_data_prepared_all_1960_1980.to_feather(output_path)

print(f"\n✅ Successfully created and saved the final DataFrame.")
print(f"   DataFrame Name: ML_data_prepared_all_1960_1980")
print(f"   Saved to path: {output_path}")
print(f"   Final DataFrame shape: {ML_data_prepared_all_1960_1980.shape}")

if __name__ == '__main__':
    pass