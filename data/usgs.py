import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.core_imports import *

USGS_DATA_URLS = {
    # 2015 (County-level, CSV from ScienceBase direct link)
    2015: {
        'url': 'https://www.sciencebase.gov/catalog/file/get/5af3311be4b0da30c1b245d8?name=usco2015v2.0.csv',
        'type': 'csv',
        'description': 'County-level water use data for 2015.',
        'key': 'usgs_2015_county',
        'skiprows': 1
    },
    # 2010 (County-level, XLSX)
    2010: {
        'url': 'https://water.usgs.gov/watuse/data/2010/usco2010.xlsx',
        'type': 'excel',
        'description': 'County-level water use data for 2010.',
        'key': 'usgs_2010_county'
        # 'skiprows': 
    },
    # 2005 (County-level, XLS - older Excel format)
    2005: {
        'url': 'https://water.usgs.gov/watuse/data/2005/usco2005.xls',
        'type': 'excel',
        'description': 'County-level water use data for 2005.',
        'key': 'usgs_2005_county'
        # 'skiprows': 1
    },
    # 2000 (County-level, XLS - older Excel format)
    2000: {
        'url': 'https://water.usgs.gov/watuse/data/2000/usco2000.xls',
        'type': 'excel',
        'description': 'County-level water use data for 2000.',
        'key': 'usgs_2000_county'
        # 'skiprows': 1
    },
    # 1995 (County-level, XLS - older Excel format)
    1995: {
        'url': 'https://water.usgs.gov/watuse/data/1995/usco1995.xls',
        'type': 'excel',
        'description': 'County-level water use data for 1995.',
        'key': 'usgs_1995_county'
        # 'skiprows': 1
    },
    # 1990 (County-level, XLS - older Excel format)
    1990: {
        'url': 'https://water.usgs.gov/watuse/data/1990/us90co.xls',
        'type': 'excel',
        'description': 'County-level water use data for 1990.',
        'key': 'usgs_1990_county'
        # 'skiprows': 1
    },
    # 1985 (County-level, TXT - space or tab-delimited text file)
    1985: {
        'url': 'https://water.usgs.gov/watuse/data/1985/us85co.txt',
        'type': 'txt',
        'description': 'County-level water use data for 1985.',
        'key': 'usgs_1985_county'
        # 'skiprows': 1
    },
    # 1950-1980 (State-level, multiple XLSX files inside a ZIP, Livestock sheet only)
    '1950-1980': {
        'url': 'https://www.sciencebase.gov/catalog/file/get/584f00cee4b0260a373819db?name=Export.1950-1980-United-States-Compilation-metadata.zip',
        'type': 'zip',
        'description': 'State-level livestock water use data (consumption, total withdrawal, and consumption-to-withdrawal ratio) for 1960, 1965, 1970, 1975, and 1980.',
        'key': 'usgs_1950_1980_state',
        'filenames_in_zip': [
            'Export.1960-United-States-Compilation.xlsx',
            'Export.1965-United-States-Compilation.xlsx',
            'Export.1970-United-States-Compilation.xlsx',
            'Export.1975-United-States-Compilation.xlsx',
            'Export.1980-United-States-Compilation.xlsx'
        ],
        'sheet_name': 'LS',
        'skiprows': 3
    }
}

def download_data(url: str) -> bytes or None:
    """Downloads data from a given URL with error handling."""
    print(f"-> Attempting to download from: {url}")
    try:
        # USGS/ScienceBase URLs can sometimes be slow, increasing timeout
        response = requests.get(url, timeout=30)
        # Check for permanent HTTP errors (4xx or 5xx)
        response.raise_for_status() 
        print("   Download successful.")
        return response.content
    except requests.exceptions.HTTPError as errh:
        print(f"   HTTP Error occurred: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"   Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"   Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"   An unexpected error occurred: {err}")
    return None

def load_dataframes() -> Dict[str, pd.DataFrame]:
    """
    Downloads and loads all specified USGS water use datasets into a dictionary 
    of pandas DataFrames, handling different file formats.
    
    NOTE ON EXCEL FILES:
    The 2000, 2005, 2010, 1995, and 1990 files are in Excel formats (.xls and .xlsx). 
    Requires 'openpyxl' for .xlsx (2010) and 'xlrd' for .xls (2000, 2005, 1995, 1990).
    The 1950-1980 data consists of multiple .xlsx files in a ZIP, using only the LS (Livestock) sheet,
    with the first three rows skipped and 'Area' renamed to 'STATE'. Two DataFrames are created:
    usgs_1950_1980_state (consumption, total withdrawal, and ratio) and 
    usgs_1950_1980_state_consumption_ratio (only ratio).
    The 1985 data is a .txt file, treated as a space- or tab-delimited CSV.
    County-level datasets (1985-2015) skip the first row to use the second row as column names.
    """
    usgs_dataframes: Dict[str, pd.DataFrame] = {}

    for year, data_info in USGS_DATA_URLS.items():
        print(f"\n--- Processing {year} data --- ({data_info['description']})")
        
        # --- Adjusted Download Logic (Simplified) ---
        url = data_info['url']
        raw_content = download_data(url)
        
        if raw_content is None:
            print(f"!!! Skipping {year} due to download failure.")
            continue
        # --- End Adjusted Download Logic ---

        data_key = data_info['key']
        df = None
        
        try:
            if data_info['type'] == 'csv':
                # Read CSV, skipping rows if specified
                skiprows = data_info.get('skiprows', 0)
                df = pd.read_csv(io.StringIO(raw_content.decode('utf-8')), skiprows=skiprows)
            
            elif data_info['type'] == 'excel':
                # Read Excel, skipping rows if specified
                skiprows = data_info.get('skiprows', 0)
                df = pd.read_excel(io.BytesIO(raw_content), skiprows=skiprows)
            
            elif data_info['type'] == 'txt':
                # Read TXT file (assumed space- or tab-delimited), skipping rows if specified
                skiprows = data_info.get('skiprows', 0)
                df = pd.read_csv(io.StringIO(raw_content.decode('utf-8')), delim_whitespace=True, skiprows=skiprows)
            
            elif data_info['type'] == 'zip':
                # Handle Zipped file containing multiple XLSX files for 1950-1980 (Livestock sheet only)
                with zipfile.ZipFile(io.BytesIO(raw_content)) as z:
                    yearly_dfs = []
                    ratio_dfs = []
                    for filename in data_info['filenames_in_zip']:
                        if filename in z.namelist():
                            with z.open(filename) as f:
                                # Read the LS sheet, skipping the first three rows
                                yearly_df = pd.read_excel(
                                    f, 
                                    sheet_name=data_info['sheet_name'], 
                                    skiprows=data_info['skiprows']
                                )
                                # Ensure required columns exist
                                required_columns = ['Area', 'LS-WGWFr', 'LS-WSWFr', 'LS-CUsFr']
                                if not all(col in yearly_df.columns for col in required_columns):
                                    print(f"   WARNING: Missing required columns in LS sheet for {filename}. Found: {yearly_df.columns.tolist()}")
                                    continue
                                # Rename 'Area' to 'STATE'
                                yearly_df = yearly_df.rename(columns={'Area': 'STATE'})
                                # Select relevant columns
                                yearly_df = yearly_df[['STATE', 'LS-WGWFr', 'LS-WSWFr', 'LS-CUsFr']].copy()
                                # Extract year from filename
                                year_match = re.search(r'Export\.(\d{4})-United-States-Compilation\.xlsx', filename)
                                if year_match:
                                    year = year_match.group(1)
                                    # Compute Total_Withdrawal and WC-to-WW
                                    yearly_df[f'{year}_Total_Withdrawal'] = yearly_df['LS-WGWFr'] + yearly_df['LS-WSWFr']
                                    yearly_df[f'{year}'] = yearly_df['LS-CUsFr'] / yearly_df[f'{year}_Total_Withdrawal'].replace(0, pd.NA)
                                    # Create DataFrame for main data (consumption, total withdrawal, ratio)
                                    main_df = yearly_df[['STATE', 'LS-CUsFr', f'{year}_Total_Withdrawal', f'{year}']].copy()
                                    main_df = main_df.rename(columns={'LS-CUsFr': f'{year}_LS-CUsFr'})
                                    yearly_dfs.append(main_df)
                                    # Create DataFrame for ratio only
                                    ratio_df = yearly_df[['STATE', f'{year}']].copy()
                                    ratio_dfs.append(ratio_df)
                                else:
                                    print(f"   WARNING: Could not extract year from filename {filename}.")
                        else:
                            print(f"   WARNING: Expected file '{filename}' not found in zip. Files found: {z.namelist()}")
                    
                    # Combine yearly DataFrames into two final DataFrames
                    if yearly_dfs:
                        # Merge main DataFrames on 'STATE'
                        df = yearly_dfs[0]
                        for yearly_df in yearly_dfs[1:]:
                            df = df.merge(yearly_df, on='STATE', how='outer')
                        # Merge ratio DataFrames on 'STATE'
                        ratio_df = ratio_dfs[0]
                        for yearly_ratio_df in ratio_dfs[1:]:
                            ratio_df = ratio_df.merge(yearly_ratio_df, on='STATE', how='outer')
                        # Store both DataFrames
                        usgs_dataframes[data_key] = df
                        usgs_dataframes['usgs_1950_1980_state_consumption_ratio'] = ratio_df.dropna()
                    else:
                        print(f"   ERROR: No valid LS sheets found for {year} in zip.")
            
            if df is not None and data_key not in ['usgs_1950_1980_state_consumption_ratio']:
                # Store the successfully loaded DataFrame (except for ratio, which is handled above)
                usgs_dataframes[data_key] = df
                print(f"   ✅ Successfully loaded {data_key}. Shape: {df.shape}")
            
        except Exception as e:
            # Enhanced error message for Excel dependencies
            if 'openpyxl' in str(e) or 'xlrd' in str(e):
                print(f"   ERROR: Failed to process Excel file for {year}. Please ensure 'openpyxl' and 'xlrd' are installed. Reason: {e}")
            else:
                print(f"   ERROR: Failed to process file for {year}. Reason: {e}")

    return usgs_dataframes

# --- Example Usage in a Notebook ---

if __name__ == '__main__':
    print("Starting USGS Water Use Data Loading Script...")
    
    # This function call will execute the downloads and parsing
    usgs_census_dataframes = load_dataframes()
    
    print("\n=============================================")
    print("Data Loading Summary")
    print("=============================================")
    
    if usgs_census_dataframes:
        print(f"Total DataFrames loaded: {len(usgs_census_dataframes)}")
        print("Available keys and initial DataFrame shapes:")
        for key, df in usgs_census_dataframes.items():
            print(f" - {key}: {df.shape}")
        
        # Example of viewing the first few rows of the 2015 and 1950-1980 data
        if 'usgs_2015_county' in usgs_census_dataframes:
            print("\n--- Head of 2015 County Data (usgs_2015_county) ---")
            print(usgs_census_dataframes['usgs_2015_county'].head())
        if 'usgs_1950_1980_state' in usgs_census_dataframes:
            print("\n--- Head of 1950-1980 State Livestock Data (usgs_1950_1980_state) ---")
            print(usgs_census_dataframes['usgs_1950_1980_state'].head())
        if 'usgs_1950_1980_state_consumption_ratio' in usgs_census_dataframes:
            print("\n--- Head of 1950-1980 State Livestock Consumption Ratio (usgs_1950_1980_state_consumption_ratio) ---")
            print(usgs_census_dataframes['usgs_1950_1980_state_consumption_ratio'].head())
    else:
        print("No dataframes were loaded successfully. Check the error messages above.")

# FIPS to state name mapping
# fips_to_state = {
#     1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
#     8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
#     12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
#     18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
#     23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
#     28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada',
#     33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 36: 'New York',
#     37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
#     42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
#     47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia',
#     53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
# }

# # Replace STATE codes with uppercase state names in all DataFrames that have a 'STATE' column
# for key, df in usgs_census_dataframes.items():
#     if 'STATE' in df.columns:
#         df['STATE'] = df['STATE'].map(fips_to_state).str.upper()

# Verify
for key, df in usgs_census_dataframes.items():
    if 'STATE' in df.columns:
        print(f"\n--- {key} ---")
        print(df[['STATE']].head())


    print(usgs_census_dataframes.keys())

# saving
base_data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\usgs"
save_dir = os.path.join(base_data_dir, "usgs_water_data_feather")

# Ensure the Feather output directory exists
try:
    os.makedirs(save_dir, exist_ok=True)
    print(f"Target Feather directory ensured: {save_dir}")
except Exception as e:
    print(f"Error creating directory {save_dir}: {e}")
    # Consider raising an error here if directory creation is critical

print("\nStarting to save DataFrames to Feather format...")

saved_files = []
# Iterate through the dictionary and save each DataFrame as a .feather file
for key, df in usgs_census_dataframes.items():
    file_name = f"{key}.feather"
    target_path = os.path.join(save_dir, file_name)
    
    try:
        # Save the DataFrame to Feather format
        if isinstance(df, pd.DataFrame):
            df.to_feather(target_path)
            saved_files.append(target_path)
            print(f"✅ Successfully saved: {file_name}")
        else:
             print(f"⚠️ Skipping {key}: Item is not a pandas DataFrame.")
             
    except ImportError:
        # This error occurred in the execution environment due to missing dependencies.
        # This message will help you if it happens on your machine.
        print("❌ CRITICAL ERROR: 'pyarrow' dependency is missing. Please install it with 'pip install pyarrow' to save to .feather format.")
        break
    except Exception as e:
        print(f"❌ ERROR saving {file_name}: {e}")

print("\nFeather saving process complete.")
print(f"All files are located in the directory: {save_dir}")