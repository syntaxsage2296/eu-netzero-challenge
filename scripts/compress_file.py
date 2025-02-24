"""Before running this script, please check that there's a `data` folder at the same level as the `scripts` folder. This folder should contain the dataset files needed for the code to work properly."""

"""Run `python scripts/compress_file.py`"""  

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "processed_dataset.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# 📌 Load your CSV file
df = pd.read_csv(file_path, low_memory=False)
non_numeric_cols = df.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save encoder for later use

# 🎯 Define Features (X) and Target (y)
target_column = "consommation_energie"  # Change this if your target variable is different
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔀 Split into Train & Test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 💾 Save as .npz
cleaned_file_path = os.path.join(data_folder, "processed_dataset.npz")
np.savez(cleaned_file_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("✅ Data saved as 'processed_dataset.npz'.")
