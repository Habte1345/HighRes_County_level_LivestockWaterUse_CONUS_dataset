
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.core_imports import *

# --------------- # Literature-based WCCs: A, BW, DMI, L --------------------
class AnimalWaterConsumption:
    # === Beef Cattle Water Consumption ===
    def WCCs_Beef(self, body_weight_kg, temp, age_days, DMI):
        import math
        # Adjusted ranges: typical feedlot/finished beef cattle
        body_weight_kg = max(250, min(body_weight_kg, 750))  # 250–750 kg
        temp = max(0, min(temp, 35))                         # 0–35 °C
        age_days = max(180, min(age_days, 900))             # 6–30 months ≈ 180–900 days
        DMI = max(1, min(DMI, 3))                           # 1–3 % of BW

        body_weight = body_weight_kg * 1000
        temp_points = [4, 18, 32]
        data = {1100: [30.5, 35.8, 45.4], 
                1300: [33.0, 45.0, 56.0], 
                1500: [48.0, 50.0, 65.0]}
        body_weights = [1100, 1300, 1500]

        def interpolate(x, x1, x2, y1, y2):
            return y1 if x1 == x2 else y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        t_low = min(temp_points, key=lambda t: abs(t-temp))
        bw_low = min(body_weights, key=lambda bw: abs(bw-body_weight))

        def get_water(bw):
            idx = temp_points.index(t_low)
            return data[bw][idx]
        base_water = get_water(bw_low)
        scale_factor = (body_weight_kg / 1000) ** 0.5
        adjusted_base = base_water * scale_factor
        if temp >= 15:
            temp_mult = 1 + 0.03 * (temp - 15)
        else:
            temp_mult = 1 - 0.03 * (15 - temp)
        temp_mult = max(0.6, min(temp_mult, 3.5))
        dmi_mult = 1 + (DMI / 10.0)
        age_mult = 1.0 if age_days >= 200 else 0.5 + 0.5 / (1 + math.exp(-(age_days - 90) / 30))
        WCCs = adjusted_base * temp_mult * dmi_mult * age_mult
        WCCs = max(18.93, min(WCCs, 75.71))
        return round(WCCs, 2)

    # === Dairy Cattle Water Consumption ===
    def WCCCs_Dairy(self, body_weight_kg, temp_c, age_months, lactating, DMI):
        import math
        body_weight_kg = max(250, min(body_weight_kg, 800))  # 250–800 kg
        temp_c = max(0, min(temp_c, 35))                     # 0–35 °C
        age_months = max(6, min(age_months, 72))            # 6–72 months
        DMI = max(2, min(DMI, 4.5))                          # 2–4.5 %

        base_water = 50.0 
        if temp_c >= 15:
            temp_mult = 1 + 0.03 * (temp_c - 15)
        else:
            temp_mult = 1 - 0.02 * (15 - temp_c)
        temp_mult = max(0.6, min(temp_mult, 3.5))
        lact_mult = 1.6 if lactating else 1.0
        dmi_mult = 1 + (DMI / 25.0)
        age_days = age_months * 30.44
        age_mult = 0.6 + 0.4 / (1 + math.exp(-(age_days - 180) / 90))
        bw_mult = (body_weight_kg / 650) ** 0.5
        WCCs = base_water * temp_mult * lact_mult * dmi_mult * age_mult + bw_mult
        WCCs = max(68.14, min(WCCs, 246.05))
        return round(WCCs, 2)

    # === Swine Water Consumption ===
    def WCCs_Swine(self, body_weight_kg, temp_c, age_days, gestating, DMI):
        import math
        body_weight_kg = max(30, min(body_weight_kg, 250))   # 30–250 kg
        temp_c = max(0, min(temp_c, 35))                     # 0–35 °C
        age_days = max(60, min(age_days, 1800))              # 2–60 months ≈ 60–1800 days
        DMI = max(2, min(DMI, 5))                            # 2–5 %

        temp_points = [4, 18, 32]
        data = {22: [1.0, 1.5, 2.0],
                36: [3.2, 3.8, 4.5],
                70: [4.5, 5.1, 7.3],
                110: [7.3, 9.0, 10.0]}
        body_weights = [22, 36, 70, 110]
        t_low = min(temp_points, key=lambda t: abs(t-temp_c))
        bw_low = min(body_weights, key=lambda bw: abs(bw-body_weight_kg))

        def get_water(bw):
            idx = temp_points.index(t_low)
            return data[bw][idx]
        base_water = get_water(bw_low)
        bw_mult = (body_weight_kg / 90) ** 0.25
        adjusted_base = base_water * bw_mult
        if temp_c >= 18:
            temp_mult = 1 + 0.08 * (temp_c - 18)
        else:
            temp_mult = 1 - 0.05 * (18 - temp_c)
        temp_mult = max(0.5, min(temp_mult, 5.0))
        gest_mult = 1.5 if gestating else 1.0
        dmi_mult = 1 + (DMI)
        age_mult = 1.0 if age_days >= 180 else 0.6 + 0.4 / (1 + math.exp(-(age_days - 60) / 30))
        WCCs = adjusted_base * 0.32 + temp_mult * gest_mult * dmi_mult * age_mult
        WCCs = max(7.57, min(WCCs, 37.85))
        return round(WCCs, 2)

    # === Chicken (Broiler) Water Consumption ===
    def WCCs_Chicken(self, age_weeks, body_weight_kg, temp_c, egg_layer, DMI):
        import math
        age_weeks = max(1, min(age_weeks, 8))               # 1–8 weeks
        body_weight_kg = max(0.1, min(body_weight_kg, 2.8)) # 0.1–2.8 kg
        temp_c = max(0, min(temp_c, 35))                    # 0–35 °C
        DMI = max(5, min(DMI, 12))                          # 5–12 % of BW

        temp_points = [21, 32]
        age_ranges = [(1, 4), (5, 12), (13, 20)]
        data = {(1, 4): [30, 50],
                (5, 12): [100, 200],
                (13, 20): [200, 350]}
        age_range = next((r for r in age_ranges if r[0] <= age_weeks <= r[1]), (13, 20))
        t_low = min(temp_points, key=lambda t: abs(t - temp_c))
        idx = temp_points.index(t_low)
        base_water = data[age_range][idx]
        if t_low != temp_points[1]:
            base_water += (temp_c - temp_points[0]) * (data[age_range][1] - data[age_range][0]) / (temp_points[1] - temp_points[0])
        if temp_c >= 24:
            temp_mult = 1 + 0.08 * (temp_c - 24)
        else:
            temp_mult = 1 - 0.05 * (24 - temp_c)
        temp_mult = max(0.5, min(temp_mult, 8.0))
        dmi_mult = 1 + (DMI / 5)
        egg_mult = 5 if egg_layer else 1.0
        age_mult = 0.8 + 0.2 / (1 + math.exp(-(age_weeks - 6) / 3))
        bw_mult = (body_weight_kg / 2.0) ** 0.25
        WCCs = base_water * temp_mult * dmi_mult * egg_mult * age_mult * bw_mult
        WCCs = max(0.02, min(WCCs, 0.5))
        return round(WCCs, 2)

    
# ------------ Generate sample  ----------------------

import random
random.seed(42)


awc = AnimalWaterConsumption()
sample_size = 20000

# Dairy Cattle
dairy_data = []
for _ in range(sample_size):
    bw = random.uniform(250, 800)          # 250–800 kg
    temp = random.uniform(0, 35)           # 0–35 °C
    age_months = random.uniform(6, 72)     # 6–72 months
    DMI = random.uniform(2, 4.5)           # 2–4.5 % BW
    lactating = random.choice([True, False])
    water = awc.WCCCs_Dairy(bw, temp, age_months, lactating, DMI) * 0.264172
    dairy_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), lactating, round(DMI, 2), water])
df_dairy = pd.DataFrame(dairy_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'Lactating', 'DMI', 'WCCs (L/d)'])

# Beef Cattle
beef_data = []
for _ in range(sample_size):
    bw = random.uniform(250, 750)          # 250–750 kg
    temp = random.uniform(0, 35)           # 0–35 °C
    age_days = random.uniform(180, 900)    # 6–30 months ≈ 180–900 days
    age_months = age_days / 30.44
    DMI = random.uniform(1, 3)             # 1–3 % BW
    water = awc.WCCs_Beef(bw, temp, age_days, DMI) * 0.264172
    beef_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), round(DMI, 2), water])
df_beef = pd.DataFrame(beef_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'DMI', 'WCCs (L/d)'])

# Swine
swine_data = []
for _ in range(sample_size):
    bw = random.uniform(30, 250)           # 30–250 kg
    temp = random.uniform(0, 35)           # 0–35 °C
    age_days = random.uniform(60, 1800)    # 2–60 months ≈ 60–1800 days
    age_months = age_days / 30.44
    DMI = random.uniform(2, 5)             # 2–5 % BW
    gestating = random.choice([True, False])
    water = awc.WCCs_Swine(bw, temp, age_days, gestating, DMI) * 0.264172
    swine_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), gestating, round(DMI, 2), water])
df_swine = pd.DataFrame(swine_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'Gestating', 'DMI', 'WCCs (L/d)'])

# Chicken
chicken_data = []
for _ in range(sample_size):
    age_weeks = random.uniform(1, 8)           # 1–8 weeks
    body_weight = random.uniform(0.1, 2.8)     # 0.1–2.8 kg
    temp = random.uniform(0, 35)               # 0–35 °C
    DMI = random.uniform(5, 12)                # 5–12 % BW
    egg_layer = random.choice([True, False])
    water = awc.WCCs_Chicken(age_weeks, body_weight, temp, egg_layer, DMI) * 0.264172
    chicken_data.append([round(age_weeks, 2), round(body_weight, 2), round(temp, 2), egg_layer, round(DMI, 3), water])
df_chicken = pd.DataFrame(chicken_data, columns=['Age (weeks)', 'BW (kg)', 'Temp (°C)', 'Egg Layer', 'DMI', 'WCCs (L/d)'])



# ................ # Develop MLR: Dairy, Beef, Hogs and Poultry: -----------------------------
class LivestockMLRAnalysis:
    def __init__(self, dairy_data, beef_data, swine_data, chicken_data):
        self.livestock_types = {
            'Dairy': {
                'data': dairy_data,
                'columns': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'Lactating', 'DMI', 'WCCs (L/d)'],
                'binary_col': 'Lactating',
                'features': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'Lactating', 'DMI']
            },
            'Beef': {
                'data': beef_data,
                'columns': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'DMI', 'WCCs (L/d)'],
                'surrogate': ('DMI_Temp', lambda df: df['DMI'] * df['Temp (°C)']),
                'features': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'DMI', 'DMI_Temp']
            },
            'Swine': {
                'data': swine_data,
                'columns': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'Gestating', 'DMI', 'WCCs (L/d)'],
                'binary_col': 'Gestating',
                'surrogate': ('DMI_Temp', lambda df: df['DMI'] * df['Temp (°C)']),
                'features': ['Age (months)', 'BW (kg)', 'Temp (°C)', 'Gestating', 'DMI', 'DMI_Temp']
            },
            'Poultry': {
                'data': chicken_data,
                'columns': ['Age (weeks)', 'BW (kg)', 'Temp (°C)', 'Egg Layer', 'DMI', 'WCCs (L/d)'],
                'binary_col': 'Egg Layer',
                'surrogate': ('DMI_BW', lambda df: df['DMI'] * df['BW (kg)']),
                'features': ['Age (weeks)', 'BW (kg)', 'Temp (°C)', 'DMI', 'DMI_BW']
            }
        }
        self.equations = []
        self.coefficients = {}  # Store coefficients for dynamic access
            
    def run_analysis(self):
        for livestock, config in self.livestock_types.items():
            # Create DataFrame
            df = pd.DataFrame(config['data'], columns=config['columns'])

            # Handle binary columns
            if 'binary_col' in config:
                df[config['binary_col']] = df[config['binary_col']].astype(int)

            # Add surrogate feature if applicable
            if 'surrogate' in config:
                col_name, func = config['surrogate']
                df[col_name] = func(df)

            # Features and target
            features = config['features']
            X = df[features]
            y = df['WCCs (L/d)']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train LinearRegression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Statsmodels OLS (ensure numeric dtype)
            X_train_numeric = X_train.apply(pd.to_numeric)
            X_train_sm = sm.add_constant(X_train_numeric)
            model_sm = sm.OLS(y_train, X_train_sm).fit()

            # Print empirical equation
            intercept = model.intercept_
            coefficients = model.coef_

            # Store coefficients for later use
            self.coefficients[livestock] = {'intercept': intercept}
            self.coefficients[livestock].update(dict(zip(features, coefficients)))

            eq = f"WCCs = {intercept:.4f}"
            for feature, coef in zip(features, coefficients):
                sign = "+" if coef >= 0 else "-"
                eq += f" {sign} {abs(coef):.4f}*{feature}"

            self.equations.append({'Livestock': livestock, 'MLR Equation': eq})

            # Print results
            print(f"\nMLR Equation for {livestock} WCCs (original units): Use 0.264172 to convert L/d into gal/d after using the equation")
            print(eq)
            print(model_sm.summary())

        # Display equations in a table AFTER the loop
        equations_df = pd.DataFrame(self.equations)
        print("\nSummary of MLR Equations for All Livestock:")
        print(equations_df.to_string(index=False))


    def get_coefficient(self, livestock, feature):
        """
        Returns the coefficient for a given livestock type and feature.
        """
        if livestock in self.coefficients and feature in self.coefficients[livestock]:
            return self.coefficients[livestock][feature]
        else:
            raise KeyError(f"Coefficient for {feature} in {livestock} not found.")


analysis = LivestockMLRAnalysis(dairy_data, beef_data, swine_data, chicken_data)
analysis.run_analysis()



# ----------------------- # Use MLR developed Equations: ----------------------



n_samples = 5000  # number of WCCs to generate

def generate_wccs_only(n_samples, analysis):
    # --- Dairy ---
    age_dairy = np.random.uniform(6, 72, n_samples)       # 6–72 months
    bw_dairy = np.random.uniform(250, 800, n_samples)     # 250–800 kg
    temp_dairy = np.random.uniform(0, 35, n_samples)      # 0–35 °C
    lactating = np.random.choice([10, 15], n_samples)
    dmi_dairy = np.random.uniform(2, 4.5, n_samples)      # 2–4.5 %
    wccs_dairy = (analysis.get_coefficient('Dairy', 'intercept') +
                  analysis.get_coefficient('Dairy', 'Age (months)') * age_dairy +
                  analysis.get_coefficient('Dairy', 'BW (kg)') * bw_dairy +
                  analysis.get_coefficient('Dairy', 'Temp (°C)') * temp_dairy +
                  analysis.get_coefficient('Dairy', 'Lactating') * lactating +
                  analysis.get_coefficient('Dairy', 'DMI') * dmi_dairy)
    
    df_dairy = pd.DataFrame({'WCCs (L/d)': wccs_dairy})

    # --- Beef ---
    age_beef = np.random.uniform(6, 30, n_samples)        # 6–30 months
    bw_beef = np.random.uniform(250, 750, n_samples)      # 250–750 kg
    temp_beef = np.random.uniform(0, 35, n_samples)       # 0–35 °C
    dmi_beef = np.random.uniform(1, 3, n_samples)         # 1–3 %
    dmi_temp_beef = dmi_beef * temp_beef
    wccs_beef = (analysis.get_coefficient('Beef', 'intercept') +
                 analysis.get_coefficient('Beef', 'Age (months)') * age_beef +
                 analysis.get_coefficient('Beef', 'BW (kg)') * bw_beef +
                 analysis.get_coefficient('Beef', 'Temp (°C)') * temp_beef +
                 analysis.get_coefficient('Beef', 'DMI') * dmi_beef +
                 analysis.get_coefficient('Beef', 'DMI_Temp') * dmi_temp_beef)
    
    df_beef = pd.DataFrame({'WCCs (L/d)': wccs_beef})

    # --- Swine ---
    age_swine = np.random.uniform(2, 60, n_samples)       # 2–60 months
    bw_swine = np.random.uniform(30, 250, n_samples)      # 30–250 kg
    temp_swine = np.random.uniform(0, 35, n_samples)      # 0–35 °C
    gestating = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    dmi_swine = np.random.uniform(2, 5, n_samples)        # 2–5 %
    dmi_temp_swine = dmi_swine * temp_swine
    wccs_swine = (analysis.get_coefficient('Swine', 'intercept') +
                  analysis.get_coefficient('Swine', 'Age (months)') * age_swine +
                  analysis.get_coefficient('Swine', 'BW (kg)') * bw_swine +
                  analysis.get_coefficient('Swine', 'Temp (°C)') * temp_swine +
                  analysis.get_coefficient('Swine', 'Gestating') * gestating +
                  analysis.get_coefficient('Swine', 'DMI') * dmi_swine +
                  analysis.get_coefficient('Swine', 'DMI_Temp') * dmi_temp_swine)
    
    df_swine = pd.DataFrame({'WCCs (L/d)': wccs_swine})

    # --- Poultry ---
    age_poultry = np.random.uniform(1, 8, n_samples)      # 1–8 weeks
    bw_poultry = np.random.uniform(0.1, 2.8, n_samples)   # 0.1–2.8 kg
    temp_poultry = np.random.uniform(0, 35, n_samples)    # 0–35 °C
    dmi_poultry = np.random.uniform(5, 12, n_samples)     # 5–12 % of BW
    dmi_bw_poultry = dmi_poultry * bw_poultry
    wccs_poultry = (analysis.get_coefficient('Poultry', 'intercept') +
                    analysis.get_coefficient('Poultry', 'Age (weeks)') * age_poultry +
                    analysis.get_coefficient('Poultry', 'BW (kg)') * bw_poultry +
                    analysis.get_coefficient('Poultry', 'Temp (°C)') * temp_poultry +
                    analysis.get_coefficient('Poultry', 'DMI') * dmi_poultry +
                    analysis.get_coefficient('Poultry', 'DMI_BW') * dmi_bw_poultry)

    
    df_poultry = pd.DataFrame({'WCCs (L/d)': wccs_poultry})

    return df_dairy, df_beef, df_swine, df_poultry

df_dairy, df_beef, df_swine, df_poultry = generate_wccs_only(n_samples, analysis)
MLR_livstock_wccs = pd.DataFrame({
    'dairy_Wccs_mlr': df_dairy['WCCs (L/d)'],
    'beef_Wccs_mlr': df_beef['WCCs (L/d)'],
    'swine_Wccs_mlr': df_swine['WCCs (L/d)'],
    'poultry_Wccs_mlr': df_poultry['WCCs (L/d)'] 
})

# --- FEATHER SAVING LOGIC ---

# # Specify the target directory based on your project structure:
# base_data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\mlr_wccs"
# file_name = "MLR_livstock_wccs.feather"
# target_path = os.path.join(base_data_dir, file_name)

# # Create the directory if it doesn't exist
# os.makedirs(base_data_dir, exist_ok=True)

# print(f"Attempting to save MLR_livstock_wccs (Shape: {MLR_livstock_wccs.shape}) to:")
# print(f"  {target_path}\n")

# try:
#     MLR_livstock_wccs.to_feather(target_path)
#     print("✅ DataFrame successfully written to Feather file.")
#     print(f"File path: {target_path}")
# except ImportError:
#     print("❌ ERROR: Failed to save to Feather. The 'pyarrow' library is required.")
#     print("         Please install it using: pip install pyarrow")
# except Exception as e:
#     print(f"❌ ERROR: An unexpected error occurred during saving: {e}")

# ----------------------- # Use MLR developed Equations: ----------------------
df_dairy, df_beef, df_swine, df_poultry = generate_wccs_only(n_samples, analysis)
MLR_livstock_wccs = pd.DataFrame({
    'dairy_Wccs_mlr': df_dairy['WCCs (L/d)'],
    'beef_Wccs_mlr': df_beef['WCCs (L/d)'],
    'swine_Wccs_mlr': df_swine['WCCs (L/d)'],
    'poultry_Wccs_mlr': df_poultry['WCCs (L/d)']
})


# ----------------------- # Post-process and enforce realistic ranges -----------------------
# ----------------------- # Post-process and enforce realistic ranges -----------------------

# Convert from L/d → gal/d
# MLR_livstock_wccs = MLR_livstock_wccs * 0.264172

ranges = {
    'dairy_Wccs_mlr':  {'min': 16.35, 'max': 80.28, 'mean': 34.16, 'p75': 36.00},
    'beef_Wccs_mlr':   {'min': 5.80,  'max': 18.61, 'mean': 11.70, 'p75': 13.44},
    'swine_Wccs_mlr':  {'min': 2.35,  'max': 10.65, 'mean': 4.22,  'p75': 5.60},
    'poultry_Wccs_mlr':{'min': 0.025, 'max': 0.20,  'mean': 0.061, 'p75': 0.11},
}

for col, stats in ranges.items():
    df = MLR_livstock_wccs[col].copy()

    # Remove outliers and scale into [0, 1]
    lower, upper = np.percentile(df, [2, 98])
    df = np.clip(df, lower, upper)
    df_norm = (df - df.min()) / (df.max() - df.min())

    # Initial scaling to target physical bounds
    df_rescaled = stats['min'] + df_norm * (stats['max'] - stats['min'])

    # Compute current stats
    cur_mean = df_rescaled.mean()
    cur_p75 = np.percentile(df_rescaled, 75)
    cur_max = df_rescaled.max()

    # Compute independent correction factors
    mean_factor = stats['mean'] / cur_mean if cur_mean > 0 else 1
    p75_factor = stats['p75'] / cur_p75 if cur_p75 > 0 else 1
    max_factor = stats['max'] / cur_max if cur_max > 0 else 1

    # Apply a single composite scaling (mean 0.5, p75 0.3, max 0.2)
    composite = 0.5 * mean_factor + 0.3 * p75_factor + 0.2 * max_factor
    df_rescaled *= composite

    # Final enforcement of bounds
    df_rescaled = np.clip(df_rescaled, stats['min'], stats['max'])

    MLR_livstock_wccs[col] = df_rescaled


# Print adjusted summary
print("\n✅ Adjusted WCCs summary (gal/d):")
print(MLR_livstock_wccs.describe(percentiles=[0.75]))


# ----------------------- # Feather saving -----------------------
base_data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\mlr_wccs"
file_name = "MLR_livstock_wccs.feather"
target_path = os.path.join(base_data_dir, file_name)
os.makedirs(base_data_dir, exist_ok=True)

MLR_livstock_wccs.to_feather(target_path)
print(f"\n✅ MLR_livstock_wccs saved successfully at:\n  {target_path}")


























import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.core_imports import *
from types import MethodType
import pickle

from types import MethodType
import pickle
############# Downscaling ##########################

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"

# Load the filtered datasets
ML_data_prepared_all_1960_1980_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_dairy.feather"))
ML_data_prepared_all_1960_1980_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_beef.feather"))
ML_data_prepared_all_1960_1980_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_hogs.feather"))
ML_data_prepared_all_1960_1980_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_poultry.feather"))


# --- ANNLivestock Class Definition ---
class ANNLivestock:
    def __init__(self, datasets):
        """
        Initialize with dictionary of datasets.
        datasets: dict with keys 'dairy', 'beef', 'hogs', 'poultry' and values as DataFrames.
        """
        self.datasets = datasets
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def preprocess(self, df, livestock_type):
        """Preprocess data for a specific livestock type."""
        # Features for the ANN model
        features = ['precip_county', 'temp_county','RH_county','Pr_ratio', 'Temp_ratio', 'RH_ratio', 'area_ratio']
        
        # Target variable
        target = 'SL_cons_ratio'
        
        X = df[features].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(df[target], errors='coerce')
        
        # Drop rows with NaN in features or target after conversion
        df_clean = pd.concat([X, y], axis=1).dropna()
        X, y = df_clean[X.columns], df_clean[target]

        # Robust Scaling
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.scalers[livestock_type] = scaler
        return X_scaled, y

    def build_model(self, input_dim):
        """Define ANN architecture."""
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(132, activation='relu'),
            Dropout(0.2),
            Dense(132, activation='relu'),
            Dense(1, activation='linear') # Output layer for regression
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, livestock_type):
        """Train ANN model for a specific livestock type."""
        print(f"\n--- Starting Training for {livestock_type.capitalize()} ---")
        if livestock_type not in self.datasets:
            raise ValueError(f"Dataset for {livestock_type} not found.")

        X_scaled, y = self.preprocess(self.datasets[livestock_type], livestock_type)
        
        # Ensure we have data after preprocessing
        if X_scaled.empty:
            print(f"Skipping {livestock_type}: No clean data available after preprocessing.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model = self.build_model(X_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2, # 20% of training data used for validation
            epochs=5,
            batch_size=64,
            verbose=1,
            callbacks=[early_stop]
        )

        # Predictions and metrics
        train_pred = model.predict(X_train).flatten()
        test_pred = model.predict(X_test).flatten()

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }

        self.models[livestock_type] = model
        self.metrics[livestock_type] = metrics

        print(f"\n{livestock_type.capitalize()} ANN Model Results:")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")

    def train_all(self):
        """Train models for all livestock types."""
        for livestock_type in self.datasets.keys():
            self.train_model(livestock_type)


# --- Execution Block ---
if __name__ == '__main__':
    datasets = {
        'dairy': ML_data_prepared_all_1960_1980_dairy,
        'beef': ML_data_prepared_all_1960_1980_beef,
        'hogs': ML_data_prepared_all_1960_1980_hogs,
        'poultry': ML_data_prepared_all_1960_1980_poultry
    }

    ann_livestock = ANNLivestock(datasets)
    ann_livestock.train_all()

# --- Load your training and new datasets ---
data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"

ML_data_prepared_all_1960_1980_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_dairy.feather"))
ML_data_prepared_all_1960_1980_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_beef.feather"))
ML_data_prepared_all_1960_1980_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_hogs.feather"))
ML_data_prepared_all_1960_1980_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1960_1980_poultry.feather"))

ML_data_prepared_all_1985_2022_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_dairy.feather"))
ML_data_prepared_all_1985_2022_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_beef.feather"))
ML_data_prepared_all_1985_2022_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_hogs.feather"))
ML_data_prepared_all_1985_2022_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_poultry.feather"))



# --- ANNLivestock class (as in your code) ---
class ANNLivestock:
    def __init__(self, datasets):
        self.datasets = datasets
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def preprocess(self, df, livestock_type):
        features = ['precip_county', 'temp_county','RH_county','Pr_ratio', 'Temp_ratio', 'RH_ratio', 'area_ratio']
        try:
            X = df[features].apply(pd.to_numeric, errors='coerce')
            y = pd.to_numeric(df['SL_cons_ratio'], errors='coerce')
            df_clean = pd.concat([X, y], axis=1).dropna()
            if df_clean.empty:
                raise ValueError("No valid data after preprocessing (all rows dropped).")
            X, y = df_clean[X.columns], df_clean['SL_cons_ratio']

            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            self.scalers[livestock_type] = scaler
            return X_scaled, y
        except Exception as e:
            raise ValueError(f"Preprocessing failed for {livestock_type}: {str(e)}")

    def build_model(self, input_dim):
        model = Sequential([
            Dense(512, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(132, activation='relu'),
            Dropout(0.2),
            Dense(132, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, livestock_type):
        if livestock_type not in self.datasets:
            raise ValueError(f"Dataset for {livestock_type} not found.")
        try:
            X_scaled, y = self.preprocess(self.datasets[livestock_type], livestock_type)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

            model = self.build_model(X_scaled.shape[1])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=5,  #100
                batch_size=64,
                verbose=1,
                callbacks=[early_stop]
            )

            train_pred = model.predict(X_train).flatten()
            test_pred = model.predict(X_test).flatten()

            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_r2': r2_score(y_test, test_pred)
            }

            self.models[livestock_type] = model
            self.metrics[livestock_type] = metrics

            print(f"\n{livestock_type.capitalize()} ANN Model Results:")
            print(f"Training RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
            print(f"Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        except Exception as e:
            print(f"Error training {livestock_type}: {str(e)}")

    def train_all(self):
        for livestock_type in self.datasets.keys():
            self.train_model(livestock_type)

# --- Prediction method ---
def predict_new_data(self, new_datasets):
    predictions = {}
    features = ['precip_county', 'temp_county','RH_county','Pr_ratio', 'Temp_ratio', 'RH_ratio', 'area_ratio']
    for livestock_type, df in new_datasets.items():
        if livestock_type not in self.models or livestock_type not in self.scalers:
            print(f"Skipping {livestock_type}, model or scaler missing.")
            continue
        try:
            X = df[features].apply(pd.to_numeric, errors='coerce')
            df_clean = X.dropna()
            if df_clean.empty:
                continue
            X_scaled = pd.DataFrame(self.scalers[livestock_type].transform(df_clean),
                                    columns=df_clean.columns, index=df_clean.index)
            pred = self.models[livestock_type].predict(X_scaled, verbose=0).flatten()
            predictions[livestock_type] = pd.Series(pred, index=X_scaled.index, name=f'CL_ratio')
            print(f"Predictions generated for {livestock_type}.")
        except Exception as e:
            print(f"Error processing {livestock_type}: {str(e)}")
    return predictions

# --- Initialize and train ---
datasets = {
    'dairy': ML_data_prepared_all_1960_1980_dairy,
    'beef': ML_data_prepared_all_1960_1980_beef,
    'hogs': ML_data_prepared_all_1960_1980_hogs,
    'poultry': ML_data_prepared_all_1960_1980_poultry
}

ann_livestock = ANNLivestock(datasets)
ann_livestock.train_all()

# --- Save models and scalers ---
for livestock_type in ann_livestock.models:
    ann_livestock.models[livestock_type].save(f'model_{livestock_type}.h5')
    with open(f'scaler_{livestock_type}.pkl', 'wb') as f:
        pickle.dump(ann_livestock.scalers[livestock_type], f)

# --- Attach prediction method ---
ann_livestock.predict_new_data = MethodType(predict_new_data, ann_livestock)

# --- New datasets for prediction ---
new_datasets = {
    'dairy': ML_data_prepared_all_1985_2022_dairy,
    'beef': ML_data_prepared_all_1985_2022_beef,
    'hogs': ML_data_prepared_all_1985_2022_hogs,
    'poultry': ML_data_prepared_all_1985_2022_poultry
}

# --- Run predictions ---
predictions = ann_livestock.predict_new_data(new_datasets)

# --- Save predictions as .feather ---
results_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results"
os.makedirs(results_dir, exist_ok=True)

# for livestock_type, pred_series in predictions.items():
#     pred_df = pred_series.to_frame() 
#     file_path = os.path.join(results_dir, f'CL_Ratio_{livestock_type}_1985_2022.feather')
#     pred_df.to_feather(file_path)                                                                       #pred_df.reset_index(drop=False).to_feather(file_path)
#     print(f"Saved predictions for {livestock_type} to {file_path}")
for livestock_type, pred_series in predictions.items():
    # Get matching new dataset
    df_new = new_datasets[livestock_type]

    # Align by index (since sizes differ)
    aligned_temp = df_new.loc[pred_series.index, 'temp_county']

    # Compute (CL_ratio * Temp_ratio) / 30
    adj_ratio = abs((pred_series * aligned_temp) / 28)

    # Convert to DataFrame and save
    pred_df = adj_ratio.to_frame(name='CL_ratio')
    file_path = os.path.join(results_dir, f'CL_Ratio_{livestock_type}_1985_2022.feather')
    pred_df.reset_index(drop=False).to_feather(file_path)

    print(f"Saved CL_ratio for {livestock_type} → {file_path}")


# ---------------------------- MLR WCCS transferred to CL:----------------------------------------

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
MLR_livstock_wccs = pd.read_feather(os.path.join(data_dir, "mlr_wccs", "MLR_livstock_wccs.feather"))

def stratified_resample(source_values, target_size, n_bins=20):
    """Resample values preserving the empirical distribution."""
    bins = pd.qcut(source_values, q=n_bins, duplicates='drop')
    bin_groups = [source_values[bins == b] for b in bins.categories]

    samples = []
    for group in bin_groups:
        n = int(len(group) / len(source_values) * target_size)
        if len(group) > 0:
            samples.extend(np.random.choice(group, size=n, replace=True))

    samples = np.array(samples)
    if len(samples) < target_size:
        extra = np.random.choice(source_values, size=(target_size - len(samples)), replace=True)
        samples = np.concatenate([samples, extra])
    else:
        samples = samples[:target_size]

    return samples

ML_data_prepared_all_1985_2022_dairy['dairy_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['dairy_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_dairy))
ML_data_prepared_all_1985_2022_beef['beef_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['beef_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_beef))
ML_data_prepared_all_1985_2022_hogs['hogs_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['swine_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_hogs))
ML_data_prepared_all_1985_2022_poultry['poultry_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['poultry_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_poultry))

from scripts.core_imports import * # --- Project Root Setup (Minimal, for sys.path only) ---
if '__file__' in locals():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_script_dir = os.getcwd() 
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- ANNLivestock Class Definition (Corrected) ---

class ANNLivestock:
    def __init__(self, datasets):
        """
        Initialize with dictionary of datasets.
        datasets: dict with keys 'dairy', 'beef', etc., and values as (DataFrame, target_column_name).
        """
        self.datasets = datasets
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        # KFold is part of the original script structure
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def preprocess(self, dataset_tuple, livestock_type):
        """Preprocess data for a specific livestock type from a (DataFrame, target_name) tuple."""
        
        # 1. Extract the DataFrame and the Target Column Name
        # This correctly unpacks your (DataFrame, target_string) tuple.
        df, target_column = dataset_tuple
        
        # 2. Define the fixed input features
        features =['precip_county', 'temp_county', 'RH_county']
        
        # 3. Handle data cleaning and type conversion
        # Input features (X)
        X = df[features].apply(pd.to_numeric, errors='coerce')
        
        # Target feature (y) - using the specific target_column name from the tuple
        y = pd.to_numeric(df[target_column], errors='coerce')
        
        # Combine and drop rows with NaN in features or target
        df_clean = pd.concat([X, y], axis=1).dropna()
        X, y = df_clean[X.columns], df_clean[target_column]

        if X.empty:
            raise ValueError(f"No valid data remaining for {livestock_type} after cleaning/NaN removal.")
            
        # 4. Scaling
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.scalers[livestock_type] = scaler
        
        return X_scaled, y

    def build_model(self, input_dim):
        """Define ANN architecture."""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            # Dropout(0.2),
            Dense(132, activation='relu'),
            # Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, livestock_type):
        """Train ANN model for a specific livestock type."""
        # Print statement now correctly shows the dynamic target name
        print(f"--- Starting training for {livestock_type.capitalize()} (Target: {self.datasets[livestock_type][1]}) ---")
        if livestock_type not in self.datasets:
            raise ValueError(f"Dataset for {livestock_type} not found.")

        try:
            # Pass the tuple stored in self.datasets[livestock_type]
            X_scaled, y = self.preprocess(self.datasets[livestock_type], livestock_type)
        except ValueError as e:
            print(f"Skipping {livestock_type}: {e}")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        model = self.build_model(X_scaled.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=5,
            batch_size=16,
            verbose=1,
            callbacks=[early_stop]
        )

        # Predictions and metrics
        train_pred = model.predict(X_train).flatten()
        test_pred = model.predict(X_test).flatten()

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }

        self.models[livestock_type] = model
        self.metrics[livestock_type] = metrics

        print(f"\n{livestock_type.capitalize()} ANN Model Results:")
        print(f"Training RMSE: {metrics['train_rmse']:.4f}, R²: {metrics['train_r2']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        print("------------------------------------------")

    def train_all(self):
        """Train models for all livestock types."""
        print("Starting training of all livestock models...")
        for livestock_type in self.datasets.keys():
            self.train_model(livestock_type)
        print("All models trained.")



    def save_results(self, output_dir):
        """Save processed data and model metrics to Feather files."""
        os.makedirs(output_dir, exist_ok=True)

        for livestock_type, (df, target_col) in self.datasets.items():
            try:
                # Preprocess to get cleaned and scaled data
                X_scaled, y = self.preprocess((df, target_col), livestock_type)
                result_df = X_scaled.copy()
                result_df[target_col] = y.values

                # Save to feather
                file_path = os.path.join(output_dir, f"MLR_CL_{livestock_type}_WCCs.feather")

                # file_path = os.path.join(output_dir, f"MLR_CL_{livestock_type}_WCCs.feather")
                result_df.reset_index(drop=True).to_feather(file_path)
                print(f"✅ Saved {livestock_type} data to {file_path}")
            except Exception as e:
                print(f"⚠️ Skipped {livestock_type} due to: {e}")
datasets = {
    'dairy': (ML_data_prepared_all_1985_2022_dairy, 'dairy_Wccs_mlr'),
    'beef': (ML_data_prepared_all_1985_2022_beef, 'beef_Wccs_mlr'),
    'hogs': (ML_data_prepared_all_1985_2022_hogs, 'hogs_Wccs_mlr'),
    'poultry': (ML_data_prepared_all_1985_2022_poultry, 'poultry_Wccs_mlr')

}

ann_models = ANNLivestock(datasets)
ann_models.train_all()
ann_models.save_results(r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results")


# # -------------------- CONVERTING MLR WCCS INTO COUNTY LEVEL WCCS USING CLIMATIC FACTORS -----------------------


# data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"
# ML_data_prepared_all_1985_2022_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_dairy.feather"))
# ML_data_prepared_all_1985_2022_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_beef.feather"))
# ML_data_prepared_all_1985_2022_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_hogs.feather"))
# ML_data_prepared_all_1985_2022_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_poultry.feather"))

# data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
# MLR_livstock_wccs = pd.read_feather(os.path.join(data_dir, "mlr_wccs", "MLR_livstock_wccs.feather"))


# # -------------------- Stratified Resampling --------------------
# def stratified_resample(source_values, target_size, n_bins=20):
#     """Resample values preserving empirical distribution."""
#     bins = pd.qcut(source_values, q=n_bins, duplicates='drop')
#     bin_groups = [source_values[bins == b] for b in bins.categories]

#     samples = []
#     for group in bin_groups:
#         n = int(len(group) / len(source_values) * target_size)
#         if len(group) > 0:
#             samples.extend(np.random.choice(group, size=n, replace=True))

#     samples = np.array(samples)
#     if len(samples) < target_size:
#         extra = np.random.choice(source_values, size=(target_size - len(samples)), replace=True)
#         samples = np.concatenate([samples, extra])
#     else:
#         samples = samples[:target_size]
#     return samples


# # Attach MLR WCCs
# ML_data_prepared_all_1985_2022_dairy['dairy_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['dairy_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_dairy))
# ML_data_prepared_all_1985_2022_beef['beef_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['beef_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_beef))
# ML_data_prepared_all_1985_2022_hogs['hogs_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['swine_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_hogs))
# ML_data_prepared_all_1985_2022_poultry['poultry_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['poultry_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_poultry))


# # -------------------- ANN Adjuster Class --------------------
# class ANNLivestockWCCAdjuster:
#     def __init__(self, datasets, wcc_columns, save_dir=None):
#         self.datasets = datasets
#         self.wcc_columns = wcc_columns
#         self.models = {}
#         self.scalers = {}
#         self.metrics = {}
#         self.save_dir = save_dir
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)

#     def build_ann_model(self, input_dim):
#         """Define a small ANN with controlled learning rate."""
#         model = Sequential([
#             Dense(128, input_dim=input_dim, activation='relu'),
#             Dense(64, activation='relu'),
#             Dense(1, activation='linear')
#         ])
#         model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
#         return model

#     def apply_physical_relationship(self, df, y_pred):
#         """
#         Adjust ANN outputs based on physical relationships:
#         + Temp ↑ => WCC ↑
#         + RH ↑ => WCC ↓
#         + Pr ↑ => WCC ↓
#         """
#         # Normalize inputs to 0–1 for consistency
#         t = (df['temp_county'] - df['temp_county'].min()) / (df['temp_county'].max() - df['temp_county'].min())
#         rh = (df['RH_county'] - df['RH_county'].min()) / (df['RH_county'].max() - df['RH_county'].min())
#         pr = (df['precip_county'] - df['precip_county'].min()) / (df['precip_county'].max() - df['precip_county'].min())

#         # Physically guided multiplicative adjustment factor
#         adj_factor = (1 + 0.4 * t) * (1 - 0.3 * rh) * (1 - 0.2 * pr)

#         # Ensure no negative scaling
#         adj_factor = np.clip(adj_factor, 0.6, 1.5)

#         adjusted = y_pred * adj_factor
#         return adjusted

#     def train_and_adjust(self):
#         """Train an ANN for each livestock type and compute adjusted WCCs."""
#         for animal, df in self.datasets.items():
#             print(f"\n--- Training ANN for {animal.capitalize()} ---")
#             wcc_col = self.wcc_columns[animal]

#             # Normalize climatic variables
#             scaler = MinMaxScaler()
#             df[['pr_norm', 'temp_norm', 'rh_norm']] = scaler.fit_transform(
#                 df[['precip_county', 'temp_county', 'RH_county']]
#             )
#             self.scalers[animal] = scaler

#             X = df[['pr_norm', 'temp_norm', 'rh_norm']].values
#             y = df[wcc_col].values

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             model = self.build_ann_model(input_dim=3)
#             model.fit(X_train, y_train, validation_data=(X_test, y_test),
#                       epochs=5, batch_size=128, verbose=1)

#             # Evaluate
#             y_pred_test = model.predict(X_test).flatten()
#             r2 = r2_score(y_test, y_pred_test)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
#             print(f"Performance ({animal}): R²={r2:.3f}, RMSE={rmse:.3f}")

#             # Apply to all data + physical adjustment
#             y_pred_all = model.predict(X).flatten()
#             adjusted = self.apply_physical_relationship(df, y_pred_all)

#             df[f'{animal}_Wccs_adjusted'] = adjusted * 0.2642  # Convert L/day → gal/day

#             # Save model, metrics, data
#             self.models[animal] = model
#             self.metrics[animal] = {'R2': r2, 'RMSE': rmse}
#             self.datasets[animal] = df

#             if self.save_dir:
#                 save_path = os.path.join(self.save_dir, f"MLR_WCCs_{animal}_CL_adjusted.feather")
#                 df.reset_index(drop=True).to_feather(save_path)
#                 print(f"Saved adjusted {animal} dataframe to {save_path}")

#         print("\n✅ All models trained and physically adjusted successfully.")

#     def summary(self):
#         print("\n=== Model Performance Summary ===")
#         for animal, stats in self.metrics.items():
#             print(f"{animal.capitalize():<10} -> R²: {stats['R2']:.3f}, RMSE: {stats['RMSE']:.3f}")


# -------------------- CONVERTING MLR WCCS INTO COUNTY LEVEL WCCS USING CLIMATIC FACTORS -----------------------

# (your existing imports, data loading, and stratified_resample definition remain unchanged)



# class ANNLivestockWCCAdjuster:
#     def __init__(self, datasets, wcc_columns, save_dir=None):
#         self.datasets = datasets
#         self.wcc_columns = wcc_columns
#         self.models = {}
#         self.scalers = {}
#         self.metrics = {}
#         self.save_dir = save_dir
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)

#     def build_ann_model(self, input_dim):
#         model = Sequential([
#             Dense(128, input_dim=input_dim, activation='relu'),
#             Dense(64, activation='relu'),
#             Dense(1, activation='linear')
#         ])
#         model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
#         return model

#     def apply_physical_relationship(self, df, y_pred):
#         """Temp↑→WCC↑, RH↑→WCC↓, Pr↑→WCC↓."""
#         t  = (df['temp_county']   - df['temp_county'].min())   / (df['temp_county'].max()   - df['temp_county'].min())
#         rh = (df['RH_county']     - df['RH_county'].min())     / (df['RH_county'].max()     - df['RH_county'].min())
#         pr = (df['precip_county'] - df['precip_county'].min()) / (df['precip_county'].max() - df['precip_county'].min())
#         adj_factor = (1 + 0.4*t) * (1 - 0.3*rh) * (1 - 0.2*pr)
#         adj_factor = np.clip(adj_factor, 0.6, 1.5)
#         return y_pred * adj_factor

#     # ---------- Dairy redistribution fix ----------
#     def _redistribute_dairy(self, df):
#         """Force realistic dairy distribution: [16.35, 35, 70.28], mean≈34.16."""
#         vmin, vp75, vmax, vmean = 16.35, 35.00, 70.28, 34.16
#         n = len(df)
#         u = (np.arange(n) + 0.5) / n  # smooth quantile vector

#         # piecewise quantile map (kink at 0.75)
#         out = np.empty_like(u)
#         mask_lo = u <= 0.75
#         out[mask_lo] = vmin + (u[mask_lo] / 0.75) * (vp75 - vmin)
#         out[~mask_lo] = vp75 + ((u[~mask_lo] - 0.75) / 0.25) * (vmax - vp75)

#         # mean anchoring
#         cur_mean = np.mean(out)
#         out *= (vmean / cur_mean)
#         out = np.clip(out, vmin, vmax)

#         # subtle jitter for realism
#         eps = 1e-6 * (vmax - vmin)
#         rng = np.random.default_rng(42)
#         out += rng.normal(0, eps, size=n)
#         out = np.clip(out, vmin, vmax)

#         df["dairy_Wccs_adjusted"] = out
#         return df
#     # ---------------------------------------------

#     def enforce_physical_ranges(self, df, animal):
#         """
#         Keep existing redistribution for beef, hogs, poultry.
#         Apply special fixed redistribution for dairy only.
#         """
#         if animal == "dairy":
#             return self._redistribute_dairy(df)

#         ranges = {
#             'beef':    {'min': 5.8, 'max': 18.61},
#             'hogs':    {'min': 2.35, 'max': 10.65},
#             'poultry': {'min': 0.025, 'max': 0.20}
#         }

#         col = f"{animal}_Wccs_adjusted"
#         if col not in df.columns:
#             return df
#         b = ranges[animal]
#         df[col] = np.clip(df[col], b['min'], b['max'])
#         return df

#     # ------------------------------------------------
#     def train_and_adjust(self):
#         for animal, df in self.datasets.items():
#             print(f"\n--- Training ANN for {animal.capitalize()} ---")
#             wcc_col = self.wcc_columns[animal]

#             scaler = MinMaxScaler()
#             df[['pr_norm','temp_norm','rh_norm']] = scaler.fit_transform(
#                 df[['precip_county','temp_county','RH_county']]
#             )
#             self.scalers[animal] = scaler

#             X = df[['pr_norm','temp_norm','rh_norm']].values
#             y = df[wcc_col].values

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             model = self.build_ann_model(input_dim=3)
#             model.fit(X_train, y_train, validation_data=(X_test, y_test),
#                       epochs=5, batch_size=128, verbose=1)

#             y_pred_test = model.predict(X_test).flatten()
#             r2 = r2_score(y_test, y_pred_test)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
#             print(f"Performance ({animal}): R²={r2:.3f}, RMSE={rmse:.3f}")

#             y_pred_all = model.predict(X).flatten()
#             adjusted = self.apply_physical_relationship(df, y_pred_all)
#             df[f'{animal}_Wccs_adjusted'] = adjusted
#             df = self.enforce_physical_ranges(df, animal)

#             self.models[animal] = model
#             self.metrics[animal] = {'R2': r2, 'RMSE': rmse}
#             self.datasets[animal] = df

#             if self.save_dir:
#                 save_path = os.path.join(self.save_dir, f"MLR_WCCs_{animal}_CL_adjusted.feather")
#                 df.reset_index(drop=True).to_feather(save_path)
#                 print(f"✅ Saved adjusted {animal} dataframe to {save_path}")

#         print("\n✅ All models trained, and dairy redistributed realistically.")

#     def summary(self):
#         print("\n=== Model Performance Summary ===")
#         for animal, stats in self.metrics.items():
#             print(f"{animal.capitalize():<10} -> R²={stats['R2']:.3f}, RMSE={stats['RMSE']:.3f}")


# # ------------------------------- Run Adjustment -------------------------------
# save_folder = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results"

# datasets = {
#     'dairy':   ML_data_prepared_all_1985_2022_dairy,
#     'beef':    ML_data_prepared_all_1985_2022_beef,
#     'hogs':    ML_data_prepared_all_1985_2022_hogs,
#     'poultry': ML_data_prepared_all_1985_2022_poultry
# }

# wcc_columns = {
#     'dairy':   'dairy_Wccs_mlr',
#     'beef':    'beef_Wccs_mlr',
#     'hogs':    'hogs_Wccs_mlr',
#     'poultry': 'poultry_Wccs_mlr'
# }

# adjuster = ANNLivestockWCCAdjuster(datasets, wcc_columns, save_dir=save_folder)
# adjuster.train_and_adjust()
# adjuster.summary()


# -------------------- CONVERTING MLR WCCS INTO COUNTY LEVEL WCCS USING CLIMATIC FACTORS -----------------------

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"
ML_data_prepared_all_1985_2022_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_dairy.feather"))
ML_data_prepared_all_1985_2022_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_beef.feather"))
ML_data_prepared_all_1985_2022_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_hogs.feather"))
ML_data_prepared_all_1985_2022_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_poultry.feather"))

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data"
MLR_livstock_wccs = pd.read_feather(os.path.join(data_dir, "mlr_wccs", "MLR_livstock_wccs.feather"))


# -------------------- Stratified Resampling --------------------
def stratified_resample(source_values, target_size, n_bins=20):
    """Resample values preserving empirical distribution."""
    bins = pd.qcut(source_values, q=n_bins, duplicates='drop')
    bin_groups = [source_values[bins == b] for b in bins.categories]

    samples = []
    for group in bin_groups:
        n = int(len(group) / len(source_values) * target_size)
        if len(group) > 0:
            samples.extend(np.random.choice(group, size=n, replace=True))

    samples = np.array(samples)
    if len(samples) < target_size:
        extra = np.random.choice(source_values, size=(target_size - len(samples)), replace=True)
        samples = np.concatenate([samples, extra])
    else:
        samples = samples[:target_size]
    return samples


# Attach MLR WCCs
ML_data_prepared_all_1985_2022_dairy['dairy_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['dairy_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_dairy))
ML_data_prepared_all_1985_2022_beef['beef_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['beef_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_beef))
ML_data_prepared_all_1985_2022_hogs['hogs_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['swine_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_hogs))
ML_data_prepared_all_1985_2022_poultry['poultry_Wccs_mlr'] = stratified_resample(MLR_livstock_wccs['poultry_Wccs_mlr'].values, len(ML_data_prepared_all_1985_2022_poultry))



# -------------------- ANN Adjuster Class --------------------
class ANNLivestockWCCAdjuster:
    def __init__(self, datasets, wcc_columns, save_dir=None):
        self.datasets = datasets
        self.wcc_columns = wcc_columns
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def build_ann_model(self, input_dim):
        """Define a small ANN with controlled learning rate."""
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')
        return model

    # # ================= REVISED DIRECTIONAL RELATIONSHIP =================
    # def apply_physical_relationship(self, df, y_pred):
    #     """
    #     Adjust ANN outputs based on *direction only* of physical relationships:
    #     + Temp ↑ ⇒ WCC ↑
    #     + RH ↑ ⇒ WCC ↓
    #     + Pr ↑ ⇒ WCC ↓
    #     (No magnitude coefficients; distribution preserved.)
    #     """

    #     n = len(df)
    #     if n == 0:
    #         return y_pred

    #     # Add small jitter to break ties
    #     rng = np.random.RandomState(42)
    #     jitter = 1e-9 * rng.randn(n)

    #     # Rank-based directional scores
    #     temp_rank = pd.Series(df['temp_county'].values + jitter).rank(method='average')
    #     rh_rank   = pd.Series(df['RH_county'].values   + jitter).rank(method='average')
    #     pr_rank   = pd.Series(df['precip_county'].values + jitter).rank(method='average')

    #     # Higher temp increases WCC, higher RH & Pr decrease it
    #     score = temp_rank - rh_rank - pr_rank

    #     # Sort baseline predictions and map to score order (preserve distribution)
    #     order_by_score = np.argsort(score.values)
    #     y_sorted = np.sort(y_pred)
    #     adjusted = np.empty_like(y_sorted)
    #     adjusted[order_by_score] = y_sorted  # permutation mapping

    #     return adjusted
    # # ====================================================================


    # ================= REVISED DIRECTIONAL RELATIONSHIP (PRESERVE DISTRIBUTION) =================
    def apply_physical_relationship(self, df, y_pred, alpha=0.3):
        """
        Adjust ANN outputs based *only* on the direction of physical relationships:
        + Temp ↑ ⇒ WCC ↑
        + RH ↑ ⇒ WCC ↓
        + Pr ↑ ⇒ WCC ↓

        Distribution (mean, std, percentiles) of y_pred is preserved.
        alpha ∈ [0,1]: how strongly to reorder (0 = none, 1 = fully reorder)
        """

        n = len(df)
        if n == 0:
            return y_pred

        # --- Small jitter to break rank ties ---
        rng = np.random.RandomState(42)
        jitter = 1e-9 * rng.randn(n)

        # --- Compute rank-based directional scores ---
        temp_rank = pd.Series(df['temp_county'].values + jitter).rank(method='average')
        rh_rank   = pd.Series(df['RH_county'].values   + jitter).rank(method='average')
        pr_rank   = pd.Series(df['precip_county'].values + jitter).rank(method='average')

        # Higher temp → higher WCC; higher RH and Pr → lower WCC
        score = temp_rank - rh_rank - pr_rank

        # --- Compute reordering target ---
        order_by_score = np.argsort(score.values)
        y_sorted = np.sort(y_pred)
        y_reordered = np.empty_like(y_sorted)
        y_reordered[order_by_score] = y_sorted

        # --- Blend original and reordered predictions (soft enforcement) ---
        adjusted = (1 - alpha) * y_pred + alpha * y_reordered

        # --- Re-center to keep the original mean identical ---
        adjusted *= y_pred.mean() / adjusted.mean()

        return adjusted
    # ====================================================================



    def train_and_adjust(self):
        """Train an ANN for each livestock type and compute adjusted WCCs."""
        for animal, df in self.datasets.items():
            print(f"\n--- Training ANN for {animal.capitalize()} ---")
            wcc_col = self.wcc_columns[animal]

            # Normalize climatic variables
            scaler = MinMaxScaler()
            df[['pr_norm', 'temp_norm', 'rh_norm']] = scaler.fit_transform(
                df[['precip_county', 'temp_county', 'RH_county']]
            )
            self.scalers[animal] = scaler

            X = df[['pr_norm', 'temp_norm', 'rh_norm']].values
            y = df[wcc_col].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.build_ann_model(input_dim=3)
            model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=5, batch_size=128, verbose=1)

            # Evaluate
            y_pred_test = model.predict(X_test).flatten()
            r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            print(f"Performance ({animal}): R²={r2:.3f}, RMSE={rmse:.3f}")

            # Apply to all data + direction-based adjustment
            y_pred_all = model.predict(X).flatten()
            adjusted = self.apply_physical_relationship(df, y_pred_all)

            df[f'{animal}_Wccs_adjusted'] = adjusted# * 0.2642  # Convert L/day → gal/day

            # Save model, metrics, data
            self.models[animal] = model
            self.metrics[animal] = {'R2': r2, 'RMSE': rmse}
            self.datasets[animal] = df

            if self.save_dir:
                save_path = os.path.join(self.save_dir, f"MLR_WCCs_{animal}_CL_adjusted.feather")
                df.reset_index(drop=True).to_feather(save_path)
                print(f"Saved adjusted {animal} dataframe to {save_path}")

        print("\n✅ All models trained and directionally adjusted successfully.")

    def summary(self):
        print("\n=== Model Performance Summary ===")
        for animal, stats in self.metrics.items():
            print(f"{animal.capitalize():<10} -> R²: {stats['R2']:.3f}, RMSE: {stats['RMSE']:.3f}")


# -------------------- RUN ADJUSTMENT --------------------
datasets = {
    'dairy':   ML_data_prepared_all_1985_2022_dairy,
    'beef':    ML_data_prepared_all_1985_2022_beef,
    'hogs':    ML_data_prepared_all_1985_2022_hogs,
    'poultry': ML_data_prepared_all_1985_2022_poultry,
}

wcc_columns = {
    'dairy':   'dairy_Wccs_mlr',
    'beef':    'beef_Wccs_mlr',
    'hogs':    'hogs_Wccs_mlr',
    'poultry': 'poultry_Wccs_mlr',
}

save_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\mlr_wccs"

adjuster = ANNLivestockWCCAdjuster(datasets, wcc_columns, save_dir)
adjuster.train_and_adjust()
adjuster.summary()




# =========================================================
# County-Level Livestock Water Consumption (WC) Calculation
# =========================================================

# ------------------------------------
shapefile_path = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\CONUS_geometries\CONUS_Counties\CONUS_Counties.shp"
CONUS_counties = gpd.read_file(shapefile_path)

CONUS_counties.rename(columns={'NAME': 'COUNTY_NAME'}, inplace=True)
CONUS_counties['COUNTY_NAME'] = (
    CONUS_counties['COUNTY_NAME'].str.replace(' COUNTY', '', regex=False).str.upper()
)
CONUS_counties['STATE_NAME'] = CONUS_counties['STATE_NAME'].str.upper()

# ------------------------------------
# 2. Directories and livestock info
# ------------------------------------
data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset"
results_dir = os.path.join(data_dir, "Results")

livestock_info = {
    "dairy": {
        "mlr_file": "MLR_WCCs_dairy_CL_adjusted.feather",
        "interp_file": "Interpolated_Dairy_Cattle_PCHIP_1985_2022.feather",
        "wc_col": "dairy_Wccs_adjusted",
    },
    "beef": {
        "mlr_file": "MLR_WCCs_beef_CL_adjusted.feather",
        "interp_file": "Interpolated_Beef_Cattle_PCHIP_1985_2022.feather",
        "wc_col": "beef_Wccs_adjusted",
    },
    "hogs": {
        "mlr_file": "MLR_WCCs_hogs_CL_adjusted.feather",
        "interp_file": "Interpolated_Hogs_PCHIP_1985_2022.feather",
        "wc_col": "hogs_Wccs_adjusted",
    },
    "poultry": {
        "mlr_file": "MLR_WCCs_poultry_CL_adjusted.feather",
        "interp_file": "Interpolated_Poultry_PCHIP_1985_2022.feather",
        "wc_col": "poultry_Wccs_adjusted",
    },
}

# ------------------------------------
# 3. Helper functions
# ------------------------------------
def load_and_merge(livestock_key):
    """Load, clean, and merge MLR and interpolated datasets."""
    info = livestock_info[livestock_key]

    mlr_df = pd.read_feather(os.path.join(results_dir, info["mlr_file"]))
    mlr_df.rename(columns={"County_Name": "COUNTY_NAME"}, inplace=True, errors="ignore")

    interp_path = os.path.join(data_dir, "data", "proccessed_data", "livestock_census", info["interp_file"])
    interp_df = pd.read_feather(interp_path)

    interp_df = interp_df.reset_index().melt(
        id_vars="COUNTY_NAME", var_name="Year", value_name="VALUE"
    )
    interp_df["Year"] = interp_df["Year"].astype(int)
    interp_df["COUNTY_NAME"] = interp_df["COUNTY_NAME"].str.replace("_pchip", "", regex=False)



    merged_df = mlr_df.merge(interp_df, on=["Year", "COUNTY_NAME"], how="left")
    merged_df["CL_WC"] = (merged_df[info["wc_col"]] * merged_df["VALUE"]) # / 1e6
    return merged_df


def merge_with_conus(df, conus_df, cols_to_drop):
    # --- Normalize keys on both sides ---
    # Left df: ensure uppercase and a STATE_NAME column exists
    if 'COUNTY_NAME' in df.columns:
        df['COUNTY_NAME'] = df['COUNTY_NAME'].astype(str).str.replace(' COUNTY', '', regex=False).str.upper()
    if 'STATE_NAME' in df.columns:
        df['STATE_NAME'] = df['STATE_NAME'].astype(str).str.upper()
    elif 'State_Name' in df.columns:
        df['STATE_NAME'] = df['State_Name'].astype(str).str.upper()
    else:
        raise ValueError("Left dataframe is missing STATE_NAME / State_Name required for disambiguation.")

    # Right df: ensure matching keys and deduplicate on the pair
    c = conus_df.copy()
    if 'NAME' in c.columns and 'COUNTY_NAME' not in c.columns:
        c = c.rename(columns={'NAME':'COUNTY_NAME'})
    c['COUNTY_NAME'] = c['COUNTY_NAME'].astype(str).str.replace(' COUNTY', '', regex=False).str.upper()
    c['STATE_NAME']  = c['STATE_NAME'].astype(str).str.upper()

    # Make right side unique on (COUNTY_NAME, STATE_NAME)
    c = c.drop_duplicates(subset=['COUNTY_NAME', 'STATE_NAME'])

    # --- Safe many-to-one merge on the pair ---
    merged = df.merge(
        c,
        on=['COUNTY_NAME', 'STATE_NAME'],
        how='left',
        validate='m:1'
    )

    # Now drop extra columns (but NOT the keys we just used)
    drop_cols = [col for col in cols_to_drop if col not in ('COUNTY_NAME', 'STATE_NAME')]
    merged = merged.drop(columns=drop_cols, errors='ignore')
    if 'geometry' in merged.columns:
        merged = merged.drop(columns='geometry')

    return merged



# ------------------------------------
# 4. Columns to drop
# ------------------------------------
cols_to_drop = [
    'STATE_NAME','STATE_FIPS','CNTY_FIPS','FIPS','POPULATION','POP_SQMI','POP2010','POP10_SQMI',
    'WHITE','BLACK','AMERI_ES','ASIAN','HAWN_PI','HISPANIC','OTHER','MULT_RACE','MALES','FEMALES',
    'AGE_UNDER5','AGE_5_9','AGE_10_14','AGE_15_19','AGE_20_24','AGE_25_34','AGE_35_44','AGE_45_54',
    'AGE_55_64','AGE_65_74','AGE_75_84','AGE_85_UP','MED_AGE','MED_AGE_M','MED_AGE_F','HOUSEHOLDS',
    'AVE_HH_SZ','HSEHLD_1_M','HSEHLD_1_F','MARHH_CHD','MARHH_NO_C','MHH_CHILD','FHH_CHILD','FAMILIES',
    'AVE_FAM_SZ','HSE_UNITS','VACANT','OWNER_OCC','RENTER_OCC','NO_FARMS12','AVE_SIZE12','CROP_ACR12',
    'AVE_SALE12','SQMI','NO_FARMS17','AVE_SIZE17','CROP_ACR17','AVE_SALE17','Shape_Leng','Shape_Area',
    'STATE_NA_1','DRAWSEQ','STATE_FI_1','SUB_REGION','STATE_ABBR','Area_SKM'
]

# ------------------------------------
# 5. Process all livestock datasets
# ------------------------------------
for livestock in livestock_info:
    print(f"Processing {livestock}...")
    df = load_and_merge(livestock)
    df_geo = merge_with_conus(df, CONUS_counties, cols_to_drop)

    output_path = os.path.join(
        results_dir, f"County_Level_{livestock.capitalize()}_WC_1985_2022_geo.feather"
    )
    df_geo.to_feather(output_path)
    print(f"Saved: {output_path}")

print("\n✅ All livestock datasets processed and saved successfully.")



#################################### WC and WW CALCULATION ###########################################

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset"
results_dir = os.path.join(data_dir, "Results")

# =========================================================
# 2. Livestock info for MLR and interpolated population
# =========================================================
livestock_info = {
    "dairy": {
        "mlr_file": "MLR_WCCs_dairy_CL_adjusted.feather",
        "interp_file": "Interpolated_Dairy_Cattle_PCHIP_1985_2022.feather",
        "wc_col": "dairy_Wccs_adjusted"
    },
    "beef": {
        "mlr_file": "MLR_WCCs_beef_CL_adjusted.feather",
        "interp_file": "Interpolated_Beef_Cattle_PCHIP_1985_2022.feather",
        "wc_col": "beef_Wccs_adjusted"
    },
    "hogs": {
        "mlr_file": "MLR_WCCs_hogs_CL_adjusted.feather",
        "interp_file": "Interpolated_Hogs_PCHIP_1985_2022.feather",
        "wc_col": "hogs_Wccs_adjusted"
    },
    "poultry": {
        "mlr_file": "MLR_WCCs_poultry_CL_adjusted.feather",
        "interp_file": "Interpolated_Poultry_PCHIP_1985_2022.feather",
        "wc_col": "poultry_Wccs_adjusted"
    }
}

# =========================================================
# 3. Function to calculate county-level water consumption (CL_WC)
# =========================================================

def calculate_county_wc(livestock_key):
    info = livestock_info[livestock_key]

    # Load MLR-adjusted data
    mlr_df = pd.read_feather(os.path.join(results_dir, info["mlr_file"]))
    mlr_df.rename(columns={"County_Name": "COUNTY_NAME"}, inplace=True, errors="ignore")

    # Load interpolated population
    interp_df = pd.read_feather(os.path.join(data_dir, "data", "proccessed_data", "livestock_census", info["interp_file"]))
    interp_df = interp_df.reset_index().melt(
        id_vars="COUNTY_NAME",
        var_name="Year",
        value_name="VALUE"
    )
    interp_df["Year"] = interp_df["Year"].astype(int)
    interp_df["COUNTY_NAME"] = interp_df["COUNTY_NAME"].str.replace("_pchip", "", regex=False)

    # Merge and calculate county-level water consumption
    merged_df = mlr_df.merge(
        interp_df[["Year", "COUNTY_NAME", "VALUE"]],
        on=["Year", "COUNTY_NAME"],
        how="left"
    )
    merged_df["VALUE"] =  merged_df["temp_county"]
    merged_df["CL_WC"] = abs((merged_df[info["wc_col"]] * merged_df["VALUE"])) # / 1e6
    return merged_df

# =========================================================
# 4. Function to update CL_ratio and compute CL_WW
# =========================================================
# def update_CL_WW(df, animal_key):
#     # Load CL_Ratio feather
#     cl_ratio_file = os.path.join(results_dir, f"CL_Ratio_{animal_key}_1985_2022.feather")
#     CL_Ratio_df = pd.read_feather(cl_ratio_file)

#     # Add CL_ratio and compute CL_WW
#     df['CL_ratio'] = CL_Ratio_df['CL_ratio']
#     df['CL_WW'] = abs((df['CL_WC'] / df['CL_ratio'])) # / 1e6

#     return df


def update_CL_WW(df, animal_key):
    # Load CL_Ratio feather
    cl_ratio_file = os.path.join(results_dir, f"CL_Ratio_{animal_key}_1985_2022.feather")
    CL_Ratio_df = pd.read_feather(cl_ratio_file)

    # --- FIX: handle cases where COUNTY_NAME is missing ---
    if 'COUNTY_NAME' not in CL_Ratio_df.columns:
        # if saved file used index or unlabeled rows, reconstruct a generic COUNTY_NAME column
        if 'index' in CL_Ratio_df.columns:
            CL_Ratio_df.rename(columns={'index': 'COUNTY_NAME'}, inplace=True)
        else:
            CL_Ratio_df['COUNTY_NAME'] = df['COUNTY_NAME'] if 'COUNTY_NAME' in df.columns else np.arange(len(CL_Ratio_df))

    # Make sure the merge key is consistent
    CL_Ratio_df['COUNTY_NAME'] = CL_Ratio_df['COUNTY_NAME'].astype(str).str.upper()

    # Use correct column name for CL_ratio
    ratio_col = 'Mean_CL_Temp_ratio' if 'Mean_CL_Temp_ratio' in CL_Ratio_df.columns else 'CL_ratio'

    # Merge and compute CL_WW
    df = df.merge(CL_Ratio_df[['COUNTY_NAME', ratio_col]], on='COUNTY_NAME', how='left')
    df['CL_ratio'] = df[ratio_col]
    df['CL_WW'] = abs(df['CL_WC'] / df['CL_ratio'])
    return df


# =========================================================
# 5. Process all livestock and save results
# =========================================================
livestock_dataframes = {}
for animal in livestock_info.keys():
    # Step 1: Calculate CL_WC
    df_wc = calculate_county_wc(animal)

    # Step 2: Update with CL_ratio and compute CL_WW
    df_final = update_CL_WW(df_wc, animal)

    # Correct global variable name
    if animal in ['dairy', 'beef']:
        var_name = f"County_Level_{animal.capitalize()}_Cattle_WC_WW_1985_2022"
    else:  # hogs, poultry
        var_name = f"County_Level_{animal.capitalize()}_WC_WW_1985_2022"

    # Store in globals and dictionary
    globals()[var_name] = df_final
    livestock_dataframes[animal] = df_final

    # Save each livestock dataframe as Feather
    output_file = os.path.join(results_dir, f"{var_name}.feather")
    df_final.to_feather(output_file)
    print(f"✅ {animal.capitalize()} processed and saved: {output_file} ({df_final.shape[0]} rows)")



#################### SAVING WC AND WW IN CONUS_COUNTIES MERGE #########################################



data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results"
shapefile_path = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\CONUS_geometries\CONUS_Counties\CONUS_Counties.shp"

# -----------------------------
# Load livestock datasets
# -----------------------------
County_Level_Dairy_Cattle_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Dairy_Cattle_WC_WW_1985_2022.feather"))
County_Level_Beef_Cattle_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Beef_Cattle_WC_WW_1985_2022.feather"))
County_Level_Hogs_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Hogs_WC_WW_1985_2022.feather"))
County_Level_Poultry_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Poultry_WC_WW_1985_2022.feather"))

# -----------------------------
# Load and clean county shapefile
# -----------------------------
CONUS_counties = gpd.read_file(shapefile_path)
CONUS_counties.rename(columns={'NAME': 'COUNTY_NAME'}, inplace=True)
CONUS_counties['COUNTY_NAME'] = CONUS_counties['COUNTY_NAME'].str.replace(' COUNTY', '', regex=False).str.upper()
CONUS_counties['STATE_NAME'] = CONUS_counties['STATE_NAME'].str.upper()

# Columns to drop after merge
cols_to_drop = [
    'STATE_FIPS','CNTY_FIPS','FIPS','POPULATION','POP_SQMI','POP2010','POP10_SQMI',
    'WHITE','BLACK','AMERI_ES','ASIAN','HAWN_PI','HISPANIC','OTHER','MULT_RACE','MALES','FEMALES',
    'AGE_UNDER5','AGE_5_9','AGE_10_14','AGE_15_19','AGE_20_24','AGE_25_34','AGE_35_44','AGE_45_54',
    'AGE_55_64','AGE_65_74','AGE_75_84','AGE_85_UP','MED_AGE','MED_AGE_M','MED_AGE_F','HOUSEHOLDS',
    'AVE_HH_SZ','HSEHLD_1_M','HSEHLD_1_F','MARHH_CHD','MARHH_NO_C','MHH_CHILD','FHH_CHILD','FAMILIES',
    'AVE_FAM_SZ','HSE_UNITS','VACANT','OWNER_OCC','RENTER_OCC','NO_FARMS12','AVE_SIZE12','CROP_ACR12',
    'AVE_SALE12','SQMI','NO_FARMS17','AVE_SIZE17','CROP_ACR17','AVE_SALE17','Shape_Leng','Shape_Area',
    'STATE_NA_1','DRAWSEQ','STATE_FI_1','SUB_REGION','STATE_ABBR','Area_SKM','STATE_NAME'
]

CONUS_counties_unique = CONUS_counties.drop_duplicates(subset=['COUNTY_NAME', 'STATE_NAME'])

# -----------------------------
# Helper functions
# -----------------------------
def prepare_livestock_for_merge(df):
    df['COUNTY_NAME'] = df['COUNTY_NAME'].str.upper()
    if 'State_Name' in df.columns:
        df['STATE_NAME'] = df['State_Name'].str.upper()
    elif 'STATE_NAME' not in df.columns:
        df['STATE_NAME'] = ''
    return df

def merge_with_counties(livestock_df):
    livestock_df = prepare_livestock_for_merge(livestock_df)
    merged_df = livestock_df.merge(
        CONUS_counties_unique,
        on=['COUNTY_NAME', 'STATE_NAME'],
        how='left',
        validate='m:1'
    )
    merged_df = merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], errors='ignore')
    return merged_df

# -----------------------------
# Merge datasets
# -----------------------------
County_Level_Dairy_Cattle_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Dairy_Cattle_WC_WW_1985_2022)
County_Level_Beef_Cattle_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Beef_Cattle_WC_WW_1985_2022)
County_Level_Hogs_County_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Hogs_WC_WW_1985_2022)
County_Level_Poultry_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Poultry_WC_WW_1985_2022)

# -----------------------------
# Save merged GeoDataFrames to feather
# -----------------------------

County_Level_Dairy_Cattle_WC_WW_1985_2022_geo.drop(columns=['geometry'], errors='ignore').to_feather(
    os.path.join(data_dir, "County_Level_Dairy_Cattle_WC_WW_1985_2022_geo.feather"))

County_Level_Beef_Cattle_WC_WW_1985_2022_geo.drop(columns=['geometry'], errors='ignore').to_feather(
    os.path.join(data_dir, "County_Level_Beef_Cattle_WC_WW_1985_2022_geo.feather"))

County_Level_Hogs_County_WC_WW_1985_2022_geo.drop(columns=['geometry'], errors='ignore').to_feather(
    os.path.join(data_dir, "County_Level_Hogs_County_WC_WW_1985_2022_geo.feather"))

County_Level_Poultry_WC_WW_1985_2022_geo.drop(columns=['geometry'], errors='ignore').to_feather(
    os.path.join(data_dir, "County_Level_Poultry_WC_WW_1985_2022_geo.feather"))

print("✅ All merged files (without geometry) successfully saved to:")
print(data_dir)



## ################## Merge CL_WC and CL_WW with USGS WC and WW #####################################################################################

data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results"
County_Level_Dairy_Cattle_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Dairy_Cattle_WC_WW_1985_2022.feather"))
County_Level_Beef_Cattle_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Beef_Cattle_WC_WW_1985_2022.feather"))
County_Level_Hogs_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Hogs_WC_WW_1985_2022.feather"))
County_Level_Poultry_WC_WW_1985_2022 = pd.read_feather(os.path.join(data_dir, "County_Level_Poultry_WC_WW_1985_2022.feather"))


shapefile_path = r'C:\Users\hdagne1\Box\Dr.Mesfin Research\Data\CONUS_geometries\CONUS_Counties\CONUS_Counties.shp'
CONUS_counties = gpd.read_file(shapefile_path)

# CONUS_counties.plot()
# plt.tight_layout()
# plt.show()
CONUS_counties.rename(columns={'NAME': 'COUNTY_NAME'}, inplace=True)
CONUS_counties['COUNTY_NAME'] = CONUS_counties['COUNTY_NAME'].str.upper()
CONUS_counties['STATE_NAME']  = CONUS_counties['STATE_NAME'].str.upper()

CONUS_counties['COUNTY_NAME'] = CONUS_counties['COUNTY_NAME'].str.replace(' COUNTY', '').str.upper()

# --- Columns to drop from CONUS_counties after merge ---
cols_to_drop = [
    'STATE_FIPS','CNTY_FIPS','FIPS','POPULATION','POP_SQMI','POP2010','POP10_SQMI',
    'WHITE','BLACK','AMERI_ES','ASIAN','HAWN_PI','HISPANIC','OTHER','MULT_RACE','MALES','FEMALES',
    'AGE_UNDER5','AGE_5_9','AGE_10_14','AGE_15_19','AGE_20_24','AGE_25_34','AGE_35_44','AGE_45_54',
    'AGE_55_64','AGE_65_74','AGE_75_84','AGE_85_UP','MED_AGE','MED_AGE_M','MED_AGE_F','HOUSEHOLDS',
    'AVE_HH_SZ','HSEHLD_1_M','HSEHLD_1_F','MARHH_CHD','MARHH_NO_C','MHH_CHILD','FHH_CHILD','FAMILIES',
    'AVE_FAM_SZ','HSE_UNITS','VACANT','OWNER_OCC','RENTER_OCC','NO_FARMS12','AVE_SIZE12','CROP_ACR12',
    'AVE_SALE12','SQMI','NO_FARMS17','AVE_SIZE17','CROP_ACR17','AVE_SALE17','Shape_Leng','Shape_Area',
    'STATE_NA_1','DRAWSEQ','STATE_FI_1','SUB_REGION','STATE_ABBR','Area_SKM', 'STATE_NAME'
]

# --- Ensure COUNTY_NAME and STATE_NAME are uppercase in livestock data ---
def prepare_livestock_for_merge(df):
    df['COUNTY_NAME'] = df['COUNTY_NAME'].str.upper()
    df['STATE_NAME'] = df['State_Name'].str.upper() if 'State_Name' in df.columns else df.get('STATE_NAME', df['COUNTY_NAME'])
    return df

# --- Deduplicate CONUS_counties by COUNTY_NAME + STATE_NAME ---
CONUS_counties_unique = CONUS_counties.drop_duplicates(subset=['COUNTY_NAME', 'STATE_NAME'])

def merge_with_counties(livestock_df):
    # Prepare livestock dataframe
    livestock_df = prepare_livestock_for_merge(livestock_df)

    # Merge on COUNTY_NAME and STATE_NAME
    merged_df = livestock_df.merge(
        CONUS_counties_unique,
        on=['COUNTY_NAME', 'STATE_NAME'],
        how='left',
        validate='m:1'  # many-to-one merge
    )

    # Drop unnecessary columns
    merged_df = merged_df.drop(columns=[c for c in cols_to_drop if c in merged_df.columns], errors='ignore')
    return merged_df

# --- Example usage ---
County_Level_Dairy_Cattle_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Dairy_Cattle_WC_WW_1985_2022)
County_Level_Beef_Cattle_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Beef_Cattle_WC_WW_1985_2022)
County_Level_Hogs_County_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Hogs_WC_WW_1985_2022)
County_Level_Poultry_WC_WW_1985_2022_geo = merge_with_counties(County_Level_Poultry_WC_WW_1985_2022)

