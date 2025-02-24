"""Before running this script, please check that there's a `data` folder at the same level as the `scripts` folder. This folder should contain the dataset files needed for the code to work properly."""

"""Run `python scripts/normalize_data.py`"""  

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "cleaned_dataset.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv(file_path, low_memory=False, dtype=str)

# Convert numerical columns to float
numerical_cols = ["consommation_energie", "surface_habitable", "annee_construction", "nombre_niveaux", 
                  "surface_verriere", "surface_baies_orientees_nord", "surface_baies_orientees_est_ouest", 
                  "surface_baies_orientees_sud", "surface_planchers_hauts_deperditifs", 
                  "surface_planchers_bas_deperditifs", "surface_parois_verticales_opaques_deperditives"]

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, force non-numeric to NaN

# Fill NaNs with 0 to avoid scaling issues
df.fillna(0, inplace=True)

# Apply MinMaxScaler to scale all numeric columns to (0-1)
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the normalized dataset
cleaned_file_path = os.path.join(data_folder, "processed_dataset.csv")
df.to_csv(cleaned_file_path, index=False)

print("✅ Normalization complete. Processed dataset saved as 'processed_dataset.csv'.")
