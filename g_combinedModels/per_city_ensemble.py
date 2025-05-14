import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# load
df = pd.read_csv("/kaggle/input/finaltry/data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

for lag in range(1, 8):
    df[f"lag_{lag}"] = df["Invoice Quantity"].shift(lag)

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday
df["temp_times_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
df["wind_pressure_ratio"] = df["wind_speed_10m"] / (df["surface_pressure"] + 1e-6)
df["Sales_Zone_Mean"] = df.groupby("Sales Zone")["Invoice Quantity"].transform("mean")
df["Sales_Location_Mean"] = df.groupby("Sales Location")["Invoice Quantity"].transform("mean")
df = df.dropna()

# store results
results = []

# iterate through each unique city
for loc in df["Sales Location"].unique():
    city_df = df[df["Sales Location"] == loc].copy()

    if len(city_df) < 100:
        continue

    X = city_df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
    y = city_df["Invoice Quantity"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # base models training
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01)
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

    lgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # predict and ensemble
    lgb_pred = lgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    ensemble_pred = 0.5 * lgb_pred + 0.3 * rf_pred + 0.2 * xgb_pred

    # evaluate
    rmse = mean_squared_error(y_test, ensemble_pred, squared=False)
    r2 = r2_score(y_test, ensemble_pred)
    results.append({"City": loc, "RMSE": rmse, "R2": r2})

# convert to dataframe
results_df = pd.DataFrame(results)
print(results_df.sort_values("RMSE"))

avg_rmse = results_df["RMSE"].mean()
avg_r2 = results_df["R2"].mean()

print(f"\nAverage RMSE across cities: {avg_rmse:.2f}")
print(f"Average R2 across cities: {avg_r2:.2f}")
