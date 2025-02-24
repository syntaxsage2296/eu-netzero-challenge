"""Before running this script, please check that there's a `data` folder at the same level as the `scripts` folder. This folder should contain the dataset files needed for the code to work properly."""

"""Run `python scripts/clean_dataset.py`"""  

import os
import pandas as pd

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "geo_dep_95.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Load the dataset
df = pd.read_csv(file_path, low_memory=False)

# Fill missing numerical values with the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Save the cleaned dataset
cleaned_file_path = os.path.join(data_folder, "testing.csv")
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}")
