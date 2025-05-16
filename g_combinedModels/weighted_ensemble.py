#weighted ensemble
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/data-final/data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday

df["Prev_Day_Sales"] = df["Invoice Quantity"].shift(1)
df["Sales_3_Days_Ago"] = df["Invoice Quantity"].shift(3)
df["7Day_MA_Sales"] = df["Invoice Quantity"].rolling(7).mean()
df["14Day_MA_Sales"] = df["Invoice Quantity"].rolling(14).mean()

df["temp_times_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
df["wind_pressure_ratio"] = df["wind_speed_10m"] / (df["surface_pressure"] + 1e-6)
df["Sales_Zone_Mean"] = df.groupby("Sales Zone")["Invoice Quantity"].transform("mean")
df["Sales_Location_Mean"] = df.groupby("Sales Location")["Invoice Quantity"].transform("mean")

df = df.dropna()

X = df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
y = df["Invoice Quantity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#train models
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01, num_leaves=20, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)

lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

#prediction of models
pred_lgb = lgb_model.predict(X_test)
pred_rf = rf_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)

#mixture of all three predictions
final_pred = (
    0.5 * pred_lgb +
    0.3 * pred_rf +
    0.2 * pred_xgb
)

rmse = mean_squared_error(y_test, final_pred, squared=False)
r2 = r2_score(y_test, final_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
