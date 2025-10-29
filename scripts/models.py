
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.core_imports import *

# # --- Data Loading ---
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
            epochs=50,
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

from types import MethodType
import pickle
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
                epochs=100,  #100
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

for livestock_type, pred_series in predictions.items():
    pred_df = pred_series.to_frame()
    file_path = os.path.join(results_dir, f'CL_Ratio_{livestock_type}_1985_2022.feather')
    abs(pred_df.to_feather(file_path)) #pred_df.reset_index(drop=False).to_feather(file_path)
    print(f"Saved predictions for {livestock_type} to {file_path}")




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
            epochs=100,
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



# -------------------- cONVERTING MLR WCCS INTO COUNTY LEVEL WCCS USING CLIAMTIC FATCORS -----------------------


data_dir = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\data\proccessed_data\livestock_census"
ML_data_prepared_all_1985_2022_dairy = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_dairy.feather"))
ML_data_prepared_all_1985_2022_beef = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_beef.feather"))
ML_data_prepared_all_1985_2022_hogs = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_hogs.feather"))
ML_data_prepared_all_1985_2022_poultry = pd.read_feather(os.path.join(data_dir, "ML_data_prepared_all_1985_2022_poultry.feather"))



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

class ANNLivestockWCCAdjuster:
    def __init__(self, datasets, wcc_columns, save_dir=None):
        """
        Initialize with livestock datasets and their corresponding WCC columns.
        datasets: dict -> {'dairy': df_dairy, 'beef': df_beef, 'hogs': df_hogs, 'poultry': df_poultry}
        wcc_columns: dict -> {'dairy': 'dairy_Wccs_mlr', 'beef': 'beef_Wccs_mlr', 'hogs': 'hogs_Wccs_mlr', 'poultry': 'poultry_Wccs_mlr'}
        save_dir: str -> folder path to save adjusted dataframes in .feather format
        """
        self.datasets = datasets
        self.wcc_columns = wcc_columns
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def build_ann_model(self, input_dim):
        """Define and return a simple ANN regression model."""
        model = Sequential([
            Dense(256, input_dim=input_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        return model

    def train_and_adjust(self):
        """Train an ANN for each livestock type and compute adjusted WCCs."""
        for animal, df in self.datasets.items():
            print(f"\n--- Training ANN for {animal.capitalize()} ---")
            wcc_col = self.wcc_columns[animal]

            # Normalize climate variables
            scaler = MinMaxScaler()
            df[['pr_norm', 'temp_norm', 'rh_norm']] = scaler.fit_transform(
                df[['precip_county', 'temp_county', 'RH_county']]
            )
            self.scalers[animal] = scaler

            # Prepare inputs and target
            X = df[['pr_norm', 'temp_norm', 'rh_norm']].values
            y = df[wcc_col].values

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Build and train ANN
            model = self.build_ann_model(input_dim=3)
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=5,
                batch_size=128,
                verbose=1
            )

            # Evaluate
            y_pred = model.predict(X_test).flatten()
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"Performance ({animal}): R²={r2:.3f}, RMSE={rmse:.3f}")

            # Apply adjustment
            df[f'{animal}_Wccs_adjusted'] = model.predict(X).flatten() * 0.2642  # * 0.2642 is to change L/D TO GAL/D

            # Store model and metrics
            self.models[animal] = model
            self.metrics[animal] = {'R2': r2, 'RMSE': rmse}
            self.datasets[animal] = df

            # Save adjusted DataFrame in .feather format
            if self.save_dir:
                save_path = os.path.join(self.save_dir, f"MLR_WCCs_{animal}_CL_adjusted.feather")
                df.reset_index(drop=True).to_feather(save_path)
                print(f"Saved adjusted {animal} dataframe to {save_path}")

        print("\n✅ All models trained and WCCs adjusted successfully.")

    def summary(self):
        """Display performance summary for all livestock."""
        print("\n=== Model Performance Summary ===")
        for animal, stats in self.metrics.items():
            print(f"{animal.capitalize():<10} -> R²: {stats['R2']:.3f}, RMSE: {stats['RMSE']:.3f}")


# -------------------------------
# Example Usage
# -------------------------------

save_folder = r"C:\Users\hdagne1\Box\Dr.Mesfin Research\Codes\HighRes_County_level_LivestockWaterUse_CONUS_dataset\Results"

datasets = {
    'dairy': ML_data_prepared_all_1985_2022_dairy,
    'beef': ML_data_prepared_all_1985_2022_beef,
    'hogs': ML_data_prepared_all_1985_2022_hogs,
    'poultry': ML_data_prepared_all_1985_2022_poultry
}

wcc_columns = {
    'dairy': 'dairy_Wccs_mlr',
    'beef': 'beef_Wccs_mlr',
    'hogs': 'hogs_Wccs_mlr',
    'poultry': 'poultry_Wccs_mlr'
}

adjuster = ANNLivestockWCCAdjuster(datasets, wcc_columns, save_dir=save_folder)
adjuster.train_and_adjust()
adjuster.summary()





# =========================================================
# County-Level Livestock Water Consumption (WC) Calculation
# =========================================================

import os
import pandas as pd
import geopandas as gpd

# ------------------------------------
# 1. Load CONUS counties shapefile
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
    merged_df["CL_WC"] = (merged_df[info["wc_col"]] * merged_df["VALUE"]) / 1e6
    return merged_df


def merge_with_conus(df, conus_df, cols_to_drop):
    """Merge with CONUS counties, drop unused and geometry columns."""
    merged = df.merge(conus_df, on="COUNTY_NAME", how="left")
    merged = merged.drop(columns=cols_to_drop, errors="ignore")
    if "geometry" in merged.columns:
        merged = merged.drop(columns="geometry")
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
    merged_df["CL_WC"] = (merged_df[info["wc_col"]] * merged_df["VALUE"]) / 1e6

    return merged_df

# =========================================================
# 4. Function to update CL_ratio and compute CL_WW
# =========================================================
def update_CL_WW(df, animal_key):
    # Load CL_Ratio feather
    cl_ratio_file = os.path.join(results_dir, f"CL_Ratio_{animal_key}_1985_2022.feather")
    CL_Ratio_df = abs(pd.read_feather(cl_ratio_file))

    # Add CL_ratio and compute CL_WW
    df['CL_ratio'] = CL_Ratio_df['CL_ratio']
    df['CL_WW'] = df['CL_WC'] / df['CL_ratio']

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
