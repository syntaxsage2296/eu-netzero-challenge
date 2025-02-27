# Net-Zero Buildings Data Challenge – France Dataset

## Overview

This project focuses on analyzing energy efficiency in French buildings and developing a machine learning model to predict energy consumption and recommend cost-effective renovation measures. The study is part of the Net-Zero Buildings Data Challenge – EU Edition and aligns with the goal of improving energy efficiency in residential and commercial structures.

## Dataset

The dataset includes building characteristics, energy consumption, and renovation history. Key features:

- Building Type: Appartement, Logements Collectifs, Maison
- Energy Consumption (kWh/m²)
- Energy Efficiency Class (A-G, N for unknown)
- Renovation Status: Whether the building has undergone renovations
- Building Elements: Various features affecting energy performance

## Exploratory Data Analysis (EDA)

EDA was performed to identify trends and key factors influencing energy efficiency.

### 1️⃣ **Energy Consumption: Renovated vs. Non-Renovated Buildings**
Renovated buildings tend to have lower energy consumption.

### 2️⃣ **Average Energy Consumption by Department**
Significant regional variations in energy consumption.

### 3️⃣ **Distribution of Energy Efficiency Classes**
Class B and D are the most common efficiency ratings.

### 4️⃣ **Building Elements Impacting Energy Consumption**
Factors such as floor and wall insulation play a major role.

## Machine Learning Model

I developed an XGBoost model optimized using Optuna hyperparameter tuning and an ensemble stacking model for improved performance.

| Model  | RMSE            | R² Score                 | MAE           |
|-----------|-------------------------|-------------------------|---------------|
| XGBoost  | 0.013   | 0.9275 | 0.0004      |
| Stacking Ensemble   | 0.0013 | 0.9227      | 0.0003 |


The stacking model combines Random Forest and XGBoost for better predictions.

## How to Run the Project 

How to Run the Project 🚀

1️⃣ **Install Dependencies**

pip install -r requirements.txt

2️⃣ **Train the Model**

python scripts/train_xgb_model.py

3️⃣ **Evaluate Model Performance**

Results will be printed in the console and saved in the models/ directory.

## Conclusion

This project provides insights into energy efficiency in French buildings and presents a machine learning-based approach for predicting energy consumption. The findings can help policymakers and homeowners make data-driven decisions on renovations and sustainability improvements.