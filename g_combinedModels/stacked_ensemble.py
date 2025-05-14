# stacked_ensemble.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

df = pd.read_csv("data_final.csv", parse_dates=["Invoice Date"])
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
df = df.dropna()

X = df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
y = df["Invoice Quantity"]

X_base, X_test, y_base, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=False)

lgb_oof = np.zeros(len(X_base))
rf_oof = np.zeros(len(X_base))
xgb_oof = np.zeros(len(X_base))

# store models for test-time prediction
lgb_models = []
rf_models = []
xgb_models = []

# train base models on folds
for train_idx, val_idx in kf.split(X_base):
    X_tr, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
    y_tr, y_val = y_base.iloc[train_idx], y_base.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01, num_leaves=20, max_depth=-1)
    lgb_model.fit(X_tr, y_tr)
    lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_models.append(lgb_model)

    # RandomForest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_tr, y_tr)
    rf_oof[val_idx] = rf_model.predict(X_val)
    rf_models.append(rf_model)

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_tr, y_tr)
    xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_models.append(xgb_model)

# Stack predictions for meta-model training
X_meta = np.vstack([lgb_oof, rf_oof, xgb_oof]).T

# train meta-model (XGBoost)
meta_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
meta_model.fit(X_meta, y_base)

# predict on test set using average of models
def average_model_preds(models, X):
    preds = np.column_stack([model.predict(X) for model in models])
    return preds.mean(axis=1)

lgb_test = average_model_preds(lgb_models, X_test)
rf_test = average_model_preds(rf_models, X_test)
xgb_test = average_model_preds(xgb_models, X_test)

# stack test predictions
X_test_meta = np.vstack([lgb_test, rf_test, xgb_test]).T
final_pred = meta_model.predict(X_test_meta)

rmse = mean_squared_error(y_test, final_pred, squared=False)
r2 = r2_score(y_test, final_pred)

print(f"Stacked RMSE: {rmse:.2f}")
print(f"Stacked R2: {r2:.2f}")
