
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.core_imports import *

# --------------- # Literature-based WCCs: A, BW, DMI, L --------------------

class AnimalWaterConsumption:
    # === Beef Cattle Water Consumption ===
    def WCCs_Beef(self, body_weight_kg, temp, age_days, DMI):
        import math
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
        # Clamp to realistic bounds: 18.93 L/d to 75.71 L/d
        WCCs = max(18.93, min(WCCs, 75.71))
        return round(WCCs, 2)

    # === Dairy Cattle Water Consumption ===
    def WCCCs_Dairy(self, body_weight_kg, temp_c, age_months, lactating, DMI):
        import math
        base_water = 50.0 
        if temp_c >= 15:
            temp_mult = 1 + 0.03 * (temp_c - 15)
        else:
            temp_mult = 1 - 0.02 * (15 - temp_c)
        temp_mult = max(0.6, min(temp_mult, 3.5))
        lact_mult = 1.6 if lactating else 1.0
        dmi_mult = 1 + (DMI / 25.0)
        age_days = age_months * 30.44  # More precise conversion
        age_mult = 0.6 + 0.4 / (1 + math.exp(-(age_days - 180) / 90))  # Adjusted for smoother transition
        bw_mult = (body_weight_kg / 650) ** 0.5
        WCCs = base_water * temp_mult * lact_mult * dmi_mult * age_mult + bw_mult
        # Clamp to realistic bounds: 68.14 L/d to 246.05 L/d
        WCCs = max(68.14, min(WCCs, 246.05))
        return round(WCCs, 2)

    # === Swine Water Consumption ===
    def WCCs_Swine(self, body_weight_kg, temp_c, age_days, gestating, DMI):
        import math
        temp_points = [4, 18, 32]
        data = {22: [1.0, 1.5, 2.0],
                36: [3.2, 3.8, 4.5],
                70: [4.5, 5.1, 7.3],
                110: [7.3, 9.0, 10.0]}
        body_weights = [22, 36, 70, 110]  # Include all keys
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
        # Clamp to realistic bounds: 7.57 L/d to 37.85 L/d
        WCCs = max(7.57, min(WCCs, 37.85))
        return round(WCCs, 2)

    # === Chicken (Broiler) Water Consumption ===
    def WCCs_Chicken(self, age_weeks, body_weight_kg, temp_c, egg_layer, DMI):
        import math
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
        age_mult = 0.8 + 0.2 / (1 + math.exp(-(age_weeks - 6) / 3))  # Adjusted for realistic scaling
        bw_mult = (body_weight_kg / 2.0) ** 0.25
        WCCs = base_water * temp_mult * dmi_mult * egg_mult * age_mult * bw_mult
        # Clamp to realistic bounds: 75.71 mL/d to 1135.62 mL/d
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
    bw = random.uniform(200, 800)  # Adjusted for realistic dairy cattle weights
    temp = random.uniform(-10, 40)  # Wider temp range
    age_months = random.uniform(6, 60)  # Realistic age range
    DMI = random.uniform(10, 30)  # Realistic DMI for dairy
    lactating = random.choice([True, False])
    water = awc.WCCCs_Dairy(bw, temp, age_months, lactating, DMI) * 0.264172 # 0.264172 is to convert l/d to gal/d
    dairy_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), lactating, round(DMI, 2), water])
df_dairy = pd.DataFrame(dairy_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'Lactating', 'DMI', 'WCCs (L/d)'])

# Beef Cattle
beef_data = []
for _ in range(sample_size):
    bw = random.uniform(350, 750)
    temp = random.uniform(-10, 40) 
    age_days = random.uniform(30, 1000)
    age_months = age_days / 30.44
    DMI = random.uniform(5, 25)
    water = awc.WCCs_Beef(bw, temp, age_days, DMI) * 0.264172
    beef_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), round(DMI, 2), water])
df_beef = pd.DataFrame(beef_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'DMI', 'WCCs (L/d)'])

# Swine
swine_data = []
for _ in range(sample_size):
    bw = random.uniform(20, 110)
    temp = random.uniform(-10, 40) 
    age_days = random.uniform(10, 360)
    age_months = age_days / 30.44
    DMI = random.uniform(0.2, 5.0)
    gestating = random.choice([True, False])
    water = awc.WCCs_Swine(bw, temp, age_days, gestating, DMI) * 0.264172
    swine_data.append([round(age_months, 2), round(bw, 2), round(temp, 2), gestating, round(DMI, 2), water])
df_swine = pd.DataFrame(swine_data, columns=['Age (months)', 'BW (kg)', 'Temp (°C)', 'Gestating', 'DMI', 'WCCs (L/d)'])

# Chicken
chicken_data = []
for _ in range(sample_size):
    age_weeks = random.uniform(1, 5)
    body_weight = random.uniform(0.1, 4.0)
    temp = random.uniform(-10, 40) 
    DMI = random.uniform(0.02, 0.2)
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

n_samples = 10000  # number of WCCs to generate

def generate_wccs_only(n_samples, analysis):
    # --- Dairy ---
    age_dairy = np.random.uniform(9, 50, n_samples)
    bw_dairy = np.random.uniform(250, 750, n_samples)
    temp_dairy = np.random.uniform(0, 30, n_samples)
    lactating = np.random.choice([10, 15], n_samples)
    dmi_dairy = np.random.uniform(2.5, 5, n_samples) # %
    wccs_dairy = (analysis.get_coefficient('Dairy', 'intercept') +
                  analysis.get_coefficient('Dairy', 'Age (months)') * age_dairy +
                  analysis.get_coefficient('Dairy', 'BW (kg)') * bw_dairy +
                  analysis.get_coefficient('Dairy', 'Temp (°C)') * temp_dairy +
                  analysis.get_coefficient('Dairy', 'Lactating') * lactating +
                  analysis.get_coefficient('Dairy', 'DMI') * dmi_dairy)
    
    df_dairy = pd.DataFrame({'WCCs (L/d)': wccs_dairy})

    # --- Beef ---
    age_beef = np.random.uniform(9, 30, n_samples)  # days to months
    bw_beef = np.random.uniform(150, 950, n_samples)
    temp_beef =np.random.uniform(0, 40, n_samples)
    dmi_beef = np.random.uniform(0.5, 3.5, n_samples)
    dmi_temp_beef = dmi_beef * temp_beef
    wccs_beef = (analysis.get_coefficient('Beef', 'intercept') +
                 analysis.get_coefficient('Beef', 'Age (months)') * age_beef +
                 analysis.get_coefficient('Beef', 'BW (kg)') * bw_beef +
                 analysis.get_coefficient('Beef', 'Temp (°C)') * temp_beef +
                 analysis.get_coefficient('Beef', 'DMI') * dmi_beef +
                 analysis.get_coefficient('Beef', 'DMI_Temp') * dmi_temp_beef)
    
    df_beef = pd.DataFrame({'WCCs (L/d)': wccs_beef})

    # --- Swine ---
    age_swine = np.random.uniform(10, 50, n_samples)  # days to months
    bw_swine = np.random.uniform(80, 135, n_samples)
    temp_swine = np.random.uniform(0, 40, n_samples)
    gestating = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    dmi_swine = np.random.uniform(0.5, 2.5, n_samples)
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
    age_poultry = np.random.uniform(1, 5, n_samples)  # weeks
    bw_poultry = np.random.uniform(1.2, 4.0, n_samples)
    temp_poultry = np.random.uniform(18, 36, n_samples)
    dmi_poultry = np.random.uniform(0.5, 2.5, n_samples)
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

# Specify the target directory based on your project structure:
base_data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\mlr_wccs"
file_name = "MLR_livstock_wccs.feather"
target_path = os.path.join(base_data_dir, file_name)

# Create the directory if it doesn't exist
os.makedirs(base_data_dir, exist_ok=True)

print(f"Attempting to save MLR_livstock_wccs (Shape: {MLR_livstock_wccs.shape}) to:")
print(f"  {target_path}\n")

try:
    MLR_livstock_wccs.to_feather(target_path)
    print("✅ DataFrame successfully written to Feather file.")
    print(f"File path: {target_path}")
except ImportError:
    print("❌ ERROR: Failed to save to Feather. The 'pyarrow' library is required.")
    print("         Please install it using: pip install pyarrow")
except Exception as e:
    print(f"❌ ERROR: An unexpected error occurred during saving: {e}")