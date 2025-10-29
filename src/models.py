
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
                epochs=100,
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
    pred_df.to_feather(file_path) #pred_df.reset_index(drop=False).to_feather(file_path)
    print(f"Saved predictions for {livestock_type} to {file_path}")
