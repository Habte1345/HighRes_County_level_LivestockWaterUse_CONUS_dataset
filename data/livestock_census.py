import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.core_imports import *

# --- Configuration ---

TARGET_DIR = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"
ANIMAL_TYPES = ['Dairy_Cattle', 'Beef_Cattle', 'Hogs', 'Poultry']
CENSUS_YEARS_INT = [2002, 2007, 2012, 2017, 2022]
CENSUS_YEARS_STR = [str(y) for y in CENSUS_YEARS_INT]
BASE_URL = 'https://www.nass.usda.gov/datasets/qs.census{}.txt.gz'

# Ensure the output directory exists
os.makedirs(TARGET_DIR, exist_ok=True)

# =========================================================
## 1. Data Loading and Cleaning
# =========================================================

census_dataframes = {}
print("Loading NASS Census data for 2002–2022...")

for year in CENSUS_YEARS_INT:
    url = BASE_URL.format(year)
    name = f'qs_census{year}'
    print(f"\nLoading {name} from {url}")
    try:
        # Load data directly from the compressed URL
        df = pd.read_csv(url, sep='\t', compression='gzip', low_memory=False)
        census_dataframes[name] = df
        print(f"✅ {name}: {len(df):,} rows loaded.")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")

print("\n--- Starting Data Cleaning and Type Conversion ---")

for name, df in census_dataframes.items():
    n0 = len(df)

    # Clean VALUE column text (remove commas, spaces, and placeholders)
    df['VALUE'] = (
        df['VALUE']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace(['(D)', '(Z)', '(NA)', '(X)', ''], np.nan)
    )

    # Convert to numeric, drop NaN and zero values
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df.dropna(subset=['VALUE'], inplace=True)
    df = df[df['VALUE'] != 0]

    census_dataframes[name] = df
    print(f"[{name}] Cleaned: {n0} → {len(df)} rows ({n0 - len(df)} dropped)")

print("\n--- Cleaning Complete ---")

# --- Column List for Dropping ---

cols_to_drop = ['PRODN_PRACTICE_DESC', 'UTIL_PRACTICE_DESC', 'DOMAINCAT_DESC', 'STATE_ANSI', 'STATE_FIPS_CODE',
               'STATE_ALPHA', 'ASD_CODE', 'ASD_DESC', 'COUNTY_ANSI', 'COUNTY_CODE','REGION_DESC', 'ZIP_5', 
               'WATERSHED_CODE', 'WATERSHED_DESC', 'CONGR_DISTRICT_CODE', 'COUNTRY_CODE','COUNTRY_NAME',
               'CV_%','AGG_LEVEL_DESC','STATISTICCAT_DESC', 'LOCATION_DESC', 'YEAR', 'FREQ_DESC', 'BEGIN_CODE', 
               'END_CODE', 'REFERENCE_PERIOD_DESC', 'WEEK_ENDING', 'LOAD_TIME']

# =========================================================
## 2. Filtering by Animal Type
# =========================================================

# --- Filtering Functions ---

def filter_USDA_CENSUS_Dairy_Cattle(df):
    """Filters for Dairy Cows Inventory at the County level."""
    return df[(df['SECTOR_DESC'] == 'ANIMALS & PRODUCTS') &
              (df['GROUP_DESC'] == 'LIVESTOCK') &
              (df['COMMODITY_DESC'] == 'CATTLE') &
              (df['CLASS_DESC'] == 'COWS, MILK') &
              (df['STATISTICCAT_DESC'] == 'INVENTORY') &
              (df['UNIT_DESC'] == 'HEAD') &
              (df['AGG_LEVEL_DESC'] == 'COUNTY')]

def filter_USDA_CENSUS_Beef_Cattle(df):
    """Filters for general Cattle Inventory at the County level."""
    return df[(df['SECTOR_DESC'] == 'ANIMALS & PRODUCTS') &
              (df['GROUP_DESC'] == 'LIVESTOCK') &
              (df['COMMODITY_DESC'] == 'CATTLE') &
              (df['STATISTICCAT_DESC'] == 'INVENTORY') &
              (df['UNIT_DESC'] == 'HEAD') &
              (df['AGG_LEVEL_DESC'] == 'COUNTY')]

def filter_USDA_CENSUS_Poultry(df):
    """Filters for Chicken Layers Inventory at the County level."""
    return df[(df['SECTOR_DESC'] == 'ANIMALS & PRODUCTS') &
              (df['GROUP_DESC'] == 'POULTRY') &
              (df['COMMODITY_DESC'] == 'CHICKENS') &
              (df['STATISTICCAT_DESC'] == 'INVENTORY') &
              (df['SHORT_DESC'] == 'CHICKENS, LAYERS - INVENTORY') &
              (df['DOMAIN_DESC'] == 'TOTAL') &
              (df['AGG_LEVEL_DESC'] == 'COUNTY')]

def filter_USDA_CENSUS_Hogs(df):
    """Filters for Hogs Inventory at the County level."""
    return df[(df['SECTOR_DESC'] == 'ANIMALS & PRODUCTS') &
              (df['GROUP_DESC'] == 'LIVESTOCK') &
              (df['COMMODITY_DESC'] == 'HOGS') &
              (df['STATISTICCAT_DESC'] == 'INVENTORY') &
              (df['SHORT_DESC'] == 'HOGS - INVENTORY') &
              (df['DOMAIN_DESC'] == 'INVENTORY OF HOGS') &
              (df['AGG_LEVEL_DESC'] == 'COUNTY')]

# --- Execution of Filtering and Dropping Columns ---
filtered_dfs = {}

for year in CENSUS_YEARS_INT:
    df = census_dataframes[f'qs_census{year}']
    
    # Dairy Cattle
    filtered_dfs[f'qs_census{year}_Dairy_Cattle'] = filter_USDA_CENSUS_Dairy_Cattle(df).sort_values(by='STATE_NAME').drop(columns=cols_to_drop, axis=1, errors='ignore')
    # Beef Cattle
    filtered_dfs[f'qs_census{year}_Beef_Cattle'] = filter_USDA_CENSUS_Beef_Cattle(df).sort_values(by='STATE_NAME').drop(columns=cols_to_drop, axis=1, errors='ignore')
    # Poultry
    filtered_dfs[f'qs_census{year}_Poultry'] = filter_USDA_CENSUS_Poultry(df).sort_values(by='STATE_NAME').drop(columns=cols_to_drop, axis=1, errors='ignore')
    # Hogs
    filtered_dfs[f'qs_census{year}_Hogs'] = filter_USDA_CENSUS_Hogs(df).sort_values(by='STATE_NAME').drop(columns=cols_to_drop, axis=1, errors='ignore')
    
print("\n--- All Animal Inventories Filtered and Cleaned ---")

# =========================================================
## 3. Aggregation (SUM) by County and State
# =========================================================

print("\n--- Aggregating Inventory Values by County and Saving Feather Files ---")

# Dictionary to hold the aggregated DataFrames
aggregated_dfs = {}

for animal in ANIMAL_TYPES:
    for year in CENSUS_YEARS_INT:
        df_name_in = f'qs_census{year}_{animal}'
        df_name_out = f'qs_census{year}_{animal}_sum_value'
        
        if df_name_in in filtered_dfs:
            df = filtered_dfs[df_name_in].groupby(['STATE_NAME', 'COUNTY_NAME'], as_index=False)['VALUE'].sum()
            aggregated_dfs[df_name_out] = df
            
            # Save to feather (Intermediate step)
            file_name = f"{df_name_in}_sum.feather"
            target_path = os.path.join(TARGET_DIR, file_name)
            df.to_feather(target_path)
            print(f"✅ Saved intermediate {file_name}")

# Re-assigning to the specific variable names required in the compilation step
qs_census2002_Dairy_Cattle_sum_value = aggregated_dfs['qs_census2002_Dairy_Cattle_sum_value']
qs_census2007_Dairy_Cattle_sum_value = aggregated_dfs['qs_census2007_Dairy_Cattle_sum_value']
qs_census2012_Dairy_Cattle_sum_value = aggregated_dfs['qs_census2012_Dairy_Cattle_sum_value']
qs_census2017_Dairy_Cattle_sum_value = aggregated_dfs['qs_census2017_Dairy_Cattle_sum_value']
qs_census2022_Dairy_Cattle_sum_value = aggregated_dfs['qs_census2022_Dairy_Cattle_sum_value']

qs_census2002_Beef_Cattle_sum_value = aggregated_dfs['qs_census2002_Beef_Cattle_sum_value']
qs_census2007_Beef_Cattle_sum_value = aggregated_dfs['qs_census2007_Beef_Cattle_sum_value']
qs_census2012_Beef_Cattle_sum_value = aggregated_dfs['qs_census2012_Beef_Cattle_sum_value']
qs_census2017_Beef_Cattle_sum_value = aggregated_dfs['qs_census2017_Beef_Cattle_sum_value']
qs_census2022_Beef_Cattle_sum_value = aggregated_dfs['qs_census2022_Beef_Cattle_sum_value']

# Corrected variable names for Hogs and Poultry (removed '_Cattle')
qs_census2002_Hogs_sum_value = aggregated_dfs['qs_census2002_Hogs_sum_value']
qs_census2007_Hogs_sum_value = aggregated_dfs['qs_census2007_Hogs_sum_value']
qs_census2012_Hogs_sum_value = aggregated_dfs['qs_census2012_Hogs_sum_value']
qs_census2017_Hogs_sum_value = aggregated_dfs['qs_census2017_Hogs_sum_value']
qs_census2022_Hogs_sum_value = aggregated_dfs['qs_census2022_Hogs_sum_value']

qs_census2002_Poultry_sum_value = aggregated_dfs['qs_census2002_Poultry_sum_value']
qs_census2007_Poultry_sum_value = aggregated_dfs['qs_census2007_Poultry_sum_value']
qs_census2012_Poultry_sum_value = aggregated_dfs['qs_census2012_Poultry_sum_value']
qs_census2017_Poultry_sum_value = aggregated_dfs['qs_census2017_Poultry_sum_value']
qs_census2022_Poultry_sum_value = aggregated_dfs['qs_census2022_Poultry_sum_value']

print("\n--- All Aggregated DataFrames Saved as Feather Files ---")

# =========================================================
## 4. Compilation (Merging) and Final CSV/Feather Saving
# =========================================================

print("\n--- Compiling DataFrames (2002-2022) ---")

def compile_annual_inventory_robust(animal_type, years, dfs_dict):
    """
    Loads and merges annual inventory DataFrames (from the aggregated_dfs) 
    into a single wide-format dataframe using robust merging.
    """
    # Use the 2002 DataFrame as the base, renaming VALUE to the year
    base_df_name = f'qs_census{years[0]}_{animal_type}_sum_value'
    if base_df_name not in dfs_dict:
        print(f"❌ Base DataFrame not found for {animal_type}.")
        return None
        
    df_final = dfs_dict[base_df_name].rename(columns={'VALUE': years[0]})
    
    # Iteratively merge subsequent years
    for year in years[1:]:
        df_name_annual = f'qs_census{year}_{animal_type}_sum_value'
        if df_name_annual not in dfs_dict:
             print(f"⚠️ Annual DataFrame not found for {animal_type} year {year}. Skipping...")
             continue

        df_annual = dfs_dict[df_name_annual].rename(columns={'VALUE': year})
        
        # Robust Merge: Use State and County name to ensure correct alignment
        df_final = pd.merge(
            df_final, 
            df_annual, 
            on=['STATE_NAME', 'COUNTY_NAME'], 
            how='outer'
        )
        
    # Final cleaning and formatting
    value_cols = [str(y) for y in years]
    df_final = df_final.dropna(subset=value_cols, how='all')
    df_final[value_cols] = df_final[value_cols].fillna(0)

    # Set index to COUNTY_NAME, handling duplicates (if any) and dropping STATE_NAME
    # NOTE: County names can repeat across states, this step follows the original logic but
    # it's generally safer to keep ['STATE_NAME', 'COUNTY_NAME'] as a multi-index.
    df_final_compiled = df_final.drop(columns=['STATE_NAME']).drop_duplicates(subset=['COUNTY_NAME']).set_index('COUNTY_NAME')

    return df_final_compiled


compiled_dataframes = {}

for animal in ANIMAL_TYPES:
    compiled_df = compile_annual_inventory_robust(animal, CENSUS_YEARS_STR, aggregated_dfs)
    
    if compiled_df is not None:
        # Create the final compiled DataFrame variable names
        df_name = f'qs_census2002_{animal}_sum_value_2002_2022'
        compiled_dataframes[df_name] = compiled_df
        
        # Save the final compiled DataFrame to CSV and Feather (as requested)
        csv_file_name = f'qs_census2002_{animal}_Inventory_2002_2022.csv'
        feather_file_name = f'qs_census2002_{animal}_sum_value_2002_2022.feather'
        
        # compiled_df.to_csv(os.path.join(TARGET_DIR, csv_file_name))
        compiled_df.to_feather(os.path.join(TARGET_DIR, feather_file_name))
        
        print(f"✅ Successfully compiled and saved {feather_file_name} with {len(compiled_df)} rows.")

print("\n--- Final Compilation and Saving Complete ---")

# Re-assign to the expected variable names for the interpolation step
qs_census2002_Dairy_Cattle_sum_value_2002_2022 = compiled_dataframes.get('qs_census2002_Dairy_Cattle_sum_value_2002_2022')
qs_census2002_Beef_Cattle_sum_value_2002_2022 = compiled_dataframes.get('qs_census2002_Beef_Cattle_sum_value_2002_2022')
qs_census2002_Hogs_sum_value_2002_2022 = compiled_dataframes.get('qs_census2002_Hogs_sum_value_2002_2022')
qs_census2002_Poultry_sum_value_2002_2022 = compiled_dataframes.get('qs_census2002_Poultry_sum_value_2002_2022')


# =========================================================
## 5. Interpolation Functions
# =========================================================

def process_data_poly(df, degree=2):
    """Interpolates/extrapolates data using Polynomial Regression."""
    if df is None: return None
    
    df_T = df.T
    df_T.index = df_T.index.astype(int)
    full_index = range(1985, 2023)
    
    # Filter for census years only for fitting
    census_years_fit = [y for y in full_index if y in CENSUS_YEARS_INT]
    df_fit = df_T.loc[census_years_fit]

    years = np.array(full_index).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    years_poly = poly.fit_transform(years)
    extrapolated_df = pd.DataFrame(index=full_index, columns=df_T.columns)

    # Set bounds: min 10, max 2 * observed max
    max_values = df_fit.max()

    for county in df_T.columns:
        # Use only census year data for fitting the polynomial
        y_fit = df_fit[county].values
        
        # Fit model using only census year data points
        model = LinearRegression()
        model.fit(years_poly[[i for i, y in enumerate(full_index) if y in census_years_fit]], y_fit)
        predictions = model.predict(years_poly)
        
        # Enforce min 10 and cap at 2 * max observed value
        predictions = np.clip(predictions, 10, max_values[county] * 2)
        extrapolated_df[county] = predictions

        # Pre-2002: Linear trend based on 2002-2007 change (extrapolation for historical data)
        rate_2002 = df_fit.loc[2002, county]
        rate_2007 = df_fit.loc[2007, county]
        change_rate = (rate_2007 - rate_2002) / 5
        
        for year in range(1985, 2002):
            years_from_2002 = 2002 - year
            extrapolated_df.loc[year, county] = max(10, rate_2002 - years_from_2002 * change_rate)

    # Preserve original data points
    extrapolated_df.loc[census_years_fit] = df_fit.loc[census_years_fit]

    return extrapolated_df.T.round(0).astype(int).rename(index=lambda x: f"{x}_poly")


def process_data_spline(df):
    """Interpolates/extrapolates data using Cubic Spline Interpolation."""
    if df is None: return None

    df_T = df.T
    df_T.index = df_T.index.astype(int)
    full_years = np.arange(1985, 2023)
    extrapolated_df = pd.DataFrame(index=full_years, columns=df_T.columns)
    census_years_fit = [y for y in full_years if y in CENSUS_YEARS_INT]

    # Set bounds: min 10, max 2 * observed max
    max_values = df_T.loc[census_years_fit].max()

    for county in df_T.columns:
        x = df_T.loc[census_years_fit].index # Use only census years for spline
        y = df_T.loc[census_years_fit, county].values
        
        spline = UnivariateSpline(x, y, k=3, s=0)
        predictions = spline(full_years)
        
        # Enforce min 10 and cap at 2 * max observed value
        predictions = np.clip(predictions, 10, max_values[county] * 2)
        extrapolated_df[county] = predictions

        # Pre-2002: Linear trend based on 2002-2007 change (extrapolation for historical data)
        rate_2002 = df_T.loc[2002, county]
        rate_2007 = df_T.loc[2007, county]
        change_rate = (rate_2007 - rate_2002) / 5
        
        for year in range(1985, 2002):
            years_from_2002 = 2002 - year
            extrapolated_df.loc[year, county] = max(10, rate_2002 - years_from_2002 * change_rate)

    # Preserve original data points
    extrapolated_df.loc[census_years_fit] = df_T.loc[census_years_fit]
    return extrapolated_df.T.round(0).astype(int).rename(index=lambda x: f"{x}_spline")


def process_data_pchip(df):
    """Interpolates/extrapolates data using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)."""
    if df is None: return None

    df_T = df.T
    df_T.index = df_T.index.astype(int)
    full_years = np.arange(1985, 2023)
    extrapolated_df = pd.DataFrame(index=full_years, columns=df_T.columns)
    census_years_fit = [y for y in full_years if y in CENSUS_YEARS_INT]

    # Set bounds: min 10, max 2 * observed max
    max_values = df_T.loc[census_years_fit].max()

    for county in df_T.columns:
        x = df_T.loc[census_years_fit].index # Use only census years for PCHIP
        y = df_T.loc[census_years_fit, county].values
        
        pchip = PchipInterpolator(x, y, extrapolate=True)
        predictions = pchip(full_years)
        
        # Enforce min 10 and cap at 2 * max observed value
        predictions = np.clip(predictions, 10, max_values[county] * 2)
        extrapolated_df[county] = predictions

        # Pre-2002: Linear trend based on 2002-2007 change (extrapolation for historical data)
        rate_2002 = df_T.loc[2002, county]
        rate_2007 = df_T.loc[2007, county]
        change_rate = (rate_2007 - rate_2002) / 5
        
        for year in range(1985, 2002):
            years_from_2002 = 2002 - year
            extrapolated_df.loc[year, county] = max(10, rate_2002 - years_from_2002 * change_rate)

    # Preserve original data points
    extrapolated_df.loc[census_years_fit] = df_T.loc[census_years_fit]
    return extrapolated_df.T.round(0).astype(int).rename(index=lambda x: f"{x}_pchip")

# =========================================================
## 6. Plotting and Final Saving
# =========================================================

def plot_and_save_comparison(df, animal_type, color='m'):
    """Calculates, plots, and compares the mean population trend for different interpolation methods."""
    if df is None:
        print(f"Skipping plot and save for {animal_type}: DataFrame not loaded.")
        return

    # Process data using the three methods
    poly_df = process_data_poly(df)
    spline_df = process_data_spline(df)
    pchip_df = process_data_pchip(df)
    
    # Save the Interpolated DataFrames to Feather
    for interp_df, method in [(poly_df, 'Polynomial'), (spline_df, 'Spline'), (pchip_df, 'PCHIP')]:
        if interp_df is not None:
            feather_name = f'Interpolated_{animal_type}_{method}_1985_2022.feather'
            interp_df.to_feather(os.path.join(TARGET_DIR, feather_name))
            print(f"✅ Saved interpolated data: {feather_name}")

    # Ensure column names (county names) are strings for consistency
    for d in [poly_df, spline_df, pchip_df, df]:
        if d is not None:
            d.columns = d.columns.astype(str)

    # # Plotting
    # plt.figure(figsize=(12, 6))
    
    # if poly_df is not None:
    #     plt.plot(poly_df.mean(), label='Polynomial Regression', marker='o', color='red')
    # if spline_df is not None:
    #     plt.plot(spline_df.mean(), label='Spline Interpolation', marker='s', color='green')
    # if pchip_df is not None:
    #     plt.plot(pchip_df.mean(), label='PCHIP Interpolation', marker='^', color='blue')
        
    # # Plot original mean data (only for available years)
    # df_mean = df.T.mean(axis=1)
    # plt.plot(df_mean.index, df_mean.values, label=f'USDA {animal_type} (Observed)', linestyle='-.', marker='*', color=color, lw=2.5)

    # plt.ylabel('Mean Livestock Population (Head)')
    # plt.title(f'{animal_type.replace("_", " ")} Population Prediction (1985-2022)')
    # plt.xticks(np.arange(2002, 2023, 5), rotation=45)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

# =========================================================
## 7. Execution of Plotting and Final Saving
# =========================================================

print("\n--- Generating Interpolation Plots and Saving Final DataFrames ---")

plot_and_save_comparison(qs_census2002_Dairy_Cattle_sum_value_2002_2022, 'Dairy_Cattle', color='darkblue')
plot_and_save_comparison(qs_census2002_Beef_Cattle_sum_value_2002_2022, 'Beef_Cattle', color='brown')
plot_and_save_comparison(qs_census2002_Hogs_sum_value_2002_2022, 'Hogs', color='purple')
plot_and_save_comparison(qs_census2002_Poultry_sum_value_2002_2022, 'Poultry', color='orange')

print("\n--- All Steps Complete: Data Loaded, Processed, Compiled, Interpolated, Plotted, and Saved ---")




# ----------------- ML data prepared for each Livestock -----------------------------

# Define the base data directory
data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
livestock_census_dir = os.path.join(data_dir, "livestock_census")

# --- 1. Load the Main ML Data and Normalize County Name ---
print("1. Loading and preparing main ML data...")
ml_data_ready = os.path.join(data_dir, "ML", "ML_data_prepared_all_1960_1980.feather")
ML_data_prepared_all_1960_1980 = pd.read_feather(ml_data_ready)

# Normalize County_Name to uppercase for matching
ML_data_prepared_all_1960_1980['County_Name'] = ML_data_prepared_all_1960_1980['County_Name'].str.upper()


# --- 2. Load Livestock Census Data and Extract County Indices (County Names) ---
print("2. Loading livestock census data and extracting county names...")
# Beef Cattle
beef_path = os.path.join(livestock_census_dir, "qs_census2002_Beef_Cattle_sum_value_2002_2022.feather")
qs_census2002_Beef_Cattle_sum_value_2002_2022 = pd.read_feather(beef_path)
# Corrected line to pull from index, convert to string, and uppercase
county_beef = [str(x).upper() for x in qs_census2002_Beef_Cattle_sum_value_2002_2022.index.tolist()]

# Dairy Cattle
dairy_path = os.path.join(livestock_census_dir, "qs_census2002_Dairy_Cattle_sum_value_2002_2022.feather")
qs_census2002_Dairy_Cattle_sum_value_2002_2022 = pd.read_feather(dairy_path)
# Corrected line
county_dairy = [str(x).upper() for x in qs_census2002_Dairy_Cattle_sum_value_2002_2022.index.tolist()]

# Hogs
hogs_path = os.path.join(livestock_census_dir, "qs_census2002_Hogs_sum_value_2002_2022.feather")
qs_census2002_Hogs_sum_value_2002_2022 = pd.read_feather(hogs_path)
# Corrected line
county_hogs = [str(x).upper() for x in qs_census2002_Hogs_sum_value_2002_2022.index.tolist()]

# Poultry
poultry_path = os.path.join(livestock_census_dir, "qs_census2002_Poultry_sum_value_2002_2022.feather")
qs_census2002_Poultry_sum_value_2002_2022 = pd.read_feather(poultry_path)
# Corrected line
county_poultry = [str(x).upper() for x in qs_census2002_Poultry_sum_value_2002_2022.index.tolist()]

# --- 3. Filter the Main DataFrames by Livestock County Sets ---
print("3. Filtering main ML data by county lists...")

# Beef Cattle
ML_data_prepared_all_1960_1980_beef = ML_data_prepared_all_1960_1980[
    ML_data_prepared_all_1960_1980['County_Name'].isin(county_beef)
].reset_index(drop=True)

# Dairy Cattle
ML_data_prepared_all_1960_1980_dairy = ML_data_prepared_all_1960_1980[
    ML_data_prepared_all_1960_1980['County_Name'].isin(county_dairy)
].reset_index(drop=True)

# Hogs
ML_data_prepared_all_1960_1980_hogs = ML_data_prepared_all_1960_1980[
    ML_data_prepared_all_1960_1980['County_Name'].isin(county_hogs)
].reset_index(drop=True)

# Poultry
ML_data_prepared_all_1960_1980_poultry = ML_data_prepared_all_1960_1980[
    ML_data_prepared_all_1960_1980['County_Name'].isin(county_poultry)
].reset_index(drop=True)


# --- 4. Save the Filtered DataFrames ---
print("4. Saving filtered dataframes to .feather files...")

dataframes_to_save = {
    "ML_data_prepared_all_1960_1980_beef": ML_data_prepared_all_1960_1980_beef,
    "ML_data_prepared_all_1960_1980_dairy": ML_data_prepared_all_1960_1980_dairy,
    "ML_data_prepared_all_1960_1980_hogs": ML_data_prepared_all_1960_1980_hogs,
    "ML_data_prepared_all_1960_1980_poultry": ML_data_prepared_all_1960_1980_poultry,
}

for name, df in dataframes_to_save.items():
    output_path = os.path.join(livestock_census_dir, f"{name}.feather")
    df.to_feather(output_path)
    print(f"   - Saved {name} (Shape: {df.shape}) to:\n     {output_path}")

print("\n✅ All filtered DataFrames successfully saved.")


if __name__ == '__main__':
    pass



# ----------------------- ML data for prediction (1985-2022): ----------------------

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
livestock_census_dir = os.path.join(data_dir, "livestock_census")

# --- Load and filter climate data ---
climate_path = os.path.join(data_dir, "climatic_factors", "Annual_Climate_factors_County_Data_Ratios_and_Area_1960_2022.feather")
Annual_Climate_factors_County_Data_and_Ratios_1960_2022 = pd.read_feather(climate_path)

Annual_Climate_factors_County_Data_and_Ratios_1960_2022_filtered = (
    Annual_Climate_factors_County_Data_and_Ratios_1960_2022[
        (Annual_Climate_factors_County_Data_and_Ratios_1960_2022["Year"] >= 1985)
        & (Annual_Climate_factors_County_Data_and_Ratios_1960_2022["Year"] <= 2022)
    ]
)

ML_data_prepared_all_1985_2022 = (
    Annual_Climate_factors_County_Data_and_Ratios_1960_2022_filtered
    .sort_values(by=["State_Name", "County_Name", "Year"])
    .reset_index(drop=True)
)

ML_data_prepared_all_1985_2022['County_Name'] = ML_data_prepared_all_1985_2022['County_Name'].str.upper()

# --- Helper function to subset data by livestock type ---
def filter_by_livestock(feather_name):
    path = os.path.join(livestock_census_dir, feather_name)
    df = pd.read_feather(path)
    county_list = [str(x).upper() for x in df.index.tolist()]
    filtered = ML_data_prepared_all_1985_2022[
        ML_data_prepared_all_1985_2022['County_Name'].isin(county_list)
    ].reset_index(drop=True)
    return filtered

# --- Subset by livestock type ---
ML_data_prepared_all_1985_2022_dairy = filter_by_livestock("qs_census2002_Dairy_Cattle_sum_value_2002_2022.feather")
ML_data_prepared_all_1985_2022_beef = filter_by_livestock("qs_census2002_Beef_Cattle_sum_value_2002_2022.feather")
ML_data_prepared_all_1985_2022_hogs = filter_by_livestock("qs_census2002_Hogs_sum_value_2002_2022.feather")
ML_data_prepared_all_1985_2022_poultry = filter_by_livestock("qs_census2002_Poultry_sum_value_2002_2022.feather")

# --- Save all datasets ---
ML_data_prepared_all_1985_2022.to_feather(os.path.join(livestock_census_dir, "ML_data_prepared_all_1985_2022.feather"))
ML_data_prepared_all_1985_2022_dairy.to_feather(os.path.join(livestock_census_dir, "ML_data_prepared_all_1985_2022_dairy.feather"))
ML_data_prepared_all_1985_2022_beef.to_feather(os.path.join(livestock_census_dir, "ML_data_prepared_all_1985_2022_beef.feather"))
ML_data_prepared_all_1985_2022_hogs.to_feather(os.path.join(livestock_census_dir, "ML_data_prepared_all_1985_2022_hogs.feather"))
ML_data_prepared_all_1985_2022_poultry.to_feather(os.path.join(livestock_census_dir, "ML_data_prepared_all_1985_2022_poultry.feather"))

print("✅ All DataFrames have been successfully saved to:")
print(livestock_census_dir)