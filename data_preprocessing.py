import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load datasets
death_df = pd.read_csv("death .csv", encoding="ISO-8859-1")
incd_df = pd.read_csv("incd.csv", encoding="ISO-8859-1")

# Rename columns for consistency
death_df.columns = ["County", "FIPS", "Met_Objective", "Death_Rate", "Lower_CI_Death", "Upper_CI_Death",
                    "Annual_Deaths", "Recent_Trend_Death", "Five_Year_Trend_Death",
                    "Lower_CI_Trend_Death", "Upper_CI_Trend_Death"]

incd_df.columns = ["County", "FIPS", "Incidence_Rate", "Lower_CI", "Upper_CI",
                   "Annual_Count", "Recent_Trend", "Five_Year_Trend",
                   "Lower_CI_Trend", "Upper_CI_Trend"]

# Merge datasets on FIPS (county code)
merged_df = pd.merge(incd_df, death_df, on="FIPS")

# Drop non-numeric columns
merged_df = merged_df.drop(columns=["County_x", "County_y", "Met_Objective"])

# Convert columns to numeric
merged_df = merged_df.apply(pd.to_numeric, errors='coerce')

# Fill missing values
merged_df.fillna(merged_df.mean(), inplace=True)

# Define features and target variable (Now predicting exact Death_Rate)
X = merged_df.drop(columns=["Death_Rate"])
y = merged_df["Death_Rate"]  # 🔹 Keep Death_Rate as continuous

# Save feature names to avoid errors in prediction
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "cancer_model.pkl")

print("Model training complete! Model and feature names saved.")
