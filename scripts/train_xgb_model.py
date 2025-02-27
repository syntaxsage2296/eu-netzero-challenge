"""Before running this script, please check that there's a `data` folder at the same level as the `scripts` folder. This folder should contain the dataset files needed for the code to work properly."""

"""Run `python scripts/train_xgb_model.py`"""  

import os
import numpy as np
import xgboost as xgb
import optuna
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
model_folder = os.path.join(script_dir, ".", "models")
file_path = os.path.join(data_folder, "processed_data.npz")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# 📌 Load preprocessed dataset
data = np.load(file_path, allow_pickle=True)
X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# 🔍 Convert X_train & X_test to DataFrame for easier processing
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# 📌 1️⃣ Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 2️⃣ Remove Highly Correlated Features (Threshold = 0.9)
corr_matrix = pd.DataFrame(X_train_scaled).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features to drop
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
X_train_scaled = pd.DataFrame(X_train_scaled).drop(columns=to_drop, axis=1)
X_test_scaled = pd.DataFrame(X_test_scaled).drop(columns=to_drop, axis=1)
print(f"✅ Dropped {len(to_drop)} highly correlated features.")

# 📌 3️⃣ Remove Near-Zero Variance Features
low_variance = X_train_scaled.var()[X_train_scaled.var() < 1e-5].index
X_train_scaled = X_train_scaled.drop(columns=low_variance, axis=1)
X_test_scaled = X_test_scaled.drop(columns=low_variance, axis=1)
print(f"✅ Dropped {len(low_variance)} low-variance features.")

# 🔥 Optuna for Hyperparameter Tuning
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return mean_squared_error(y_test, y_pred)

# 🔥 Run Optuna to Find Best Hyperparameters
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("✅ Best Hyperparameters Found:", best_params)

# 🚀 Train Optimized XGBoost Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **best_params)
xgb_model.fit(X_train_scaled, y_train)

# 🎯 Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 📊 Evaluate XGBoost Model
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb) 
print(f"📉 XGBoost RMSE: {rmse_xgb:.4f}")
print(f"📈 XGBoost R² Score: {r2_xgb:.4f}")
print(f"📊 XGBoost MAE: {mae_xgb:.4f}")

# 🔥 Optimized Stacking Model
stacking_model = StackingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)),
        ("xgb", xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=42))
    ],
    final_estimator=xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, n_jobs=-1, random_state=42)
)

# 🚀 Train Optimized Stacking Model
stacking_model.fit(X_train_scaled, y_train)
y_pred_stack = stacking_model.predict(X_test_scaled)

# 📊 Evaluate Stacking Model
rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
r2_stack = r2_score(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)
print(f"📉 Optimized Stacking RMSE: {rmse_stack:.4f}")
print(f"📈 Optimized Stacking R² Score: {r2_stack:.4f}")
print(f"📊 Optimized Stacking MAE: {mae_stack:.4f}")

# 🔥 Cross-Validation for Robustness (Reduced CV folds to speed up)
cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=3, scoring="r2")
cv_scores_mae = cross_val_score(stacking_model, X_train_scaled, y_train, cv=3, scoring="neg_mean_absolute_error")

print(f"📊 Mean R² Score (Cross-Validation): {cv_scores.mean():.4f}")
print(f"📊 Mean MAE Score (Cross-Validation): {-cv_scores_mae.mean():.4f}")

# 💾 Save Final Model
cleaned_file_path = os.path.join(model_folder, "stacking_energy_model.pkl")
joblib.dump(stacking_model, cleaned_file_path)
print("💾 Model saved as 'stacking_energy_model.pkl'.")
