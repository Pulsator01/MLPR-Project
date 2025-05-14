# weighted_ensemble.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

df = pd.read_csv("data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

for lag in range(1, 8):
    df[f"lag_{lag}"] = df["Invoice Quantity"].shift(lag)
df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday
df["temp_times_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
df["wind_pressure_ratio"] = df["wind_speed_10m"] / (df["surface_pressure"] + 1e-6)
df = df.dropna()

X = df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
y = df["Invoice Quantity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 5rain models
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

lgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# ensemble
lgb_pred = lgb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

ensemble_pred = 0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * xgb_pred

print("RMSE:", mean_squared_error(y_test, ensemble_pred, squared=False))
print("R2:", r2_score(y_test, ensemble_pred))
