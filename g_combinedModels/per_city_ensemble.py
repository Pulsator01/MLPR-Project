import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb

df = pd.read_csv("/kaggle/input/data-final/data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday
df["is_weekend"] = df["Weekday"].isin([5, 6]).astype(int)

df["Prev_Day_Sales"] = df["Invoice Quantity"].shift(1)
df["Sales_3_Days_Ago"] = df["Invoice Quantity"].shift(3)
df["Sales_7_Days_Ago"] = df["Invoice Quantity"].shift(7)
df["Sales_14_Days_Ago"] = df["Invoice Quantity"].shift(14)
df["7Day_MA_Sales"] = df["Invoice Quantity"].rolling(7).mean()
df["14Day_MA_Sales"] = df["Invoice Quantity"].rolling(14).mean()
df["7Day_STD_Sales"] = df["Invoice Quantity"].rolling(7).std()

df["temp_times_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
df["wind_pressure_ratio"] = df["wind_speed_10m"] / (df["surface_pressure"] + 1e-6)
df["temp_weekday_interaction"] = df["temperature_2m"] * df["Weekday"]

df = df.dropna()
cities = df["Sales Location"].unique()
results = []

for city in cities:
    city_df = df[df["Sales Location"] == city].copy()
    if len(city_df) < 100 or city_df["Invoice Quantity"].std() < 1.0:
        continue

    X = city_df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
    y = city_df["Invoice Quantity"]
    X_base, X_test, y_base, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    lgb_oof = np.zeros(len(X_base))
    rf_oof = np.zeros(len(X_base))
    xgb_oof = np.zeros(len(X_base))
    lgb_models, rf_models, xgb_models = [], [], []
    kf = KFold(n_splits=5, shuffle=False)

    for train_idx, val_idx in kf.split(X_base):
        X_tr, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
        y_tr, y_val = y_base.iloc[train_idx], y_base.iloc[val_idx]

        lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.01, num_leaves=20, max_depth=-1)
        lgb_model.fit(X_tr, y_tr)
        lgb_oof[val_idx] = lgb_model.predict(X_val)
        lgb_models.append(lgb_model)

        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_tr, y_tr)
        rf_oof[val_idx] = rf_model.predict(X_val)
        rf_models.append(rf_model)

        xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        xgb_model.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb_model.predict(X_val)
        xgb_models.append(xgb_model)

    meta_features = ["Prev_Day_Sales", "7Day_MA_Sales", "7Day_STD_Sales", "temp_times_humidity"]
    X_meta = np.column_stack([
        lgb_oof, rf_oof, xgb_oof,
        *[X_base[feat].values for feat in meta_features]
    ])

    meta_model = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.001, early_stopping=True, max_iter=1000, random_state=42)
    meta_model.fit(X_meta, y_base)

    def predict_ensemble(models, X):
        return np.column_stack([m.predict(X) for m in models]).mean(axis=1)

    lgb_test = predict_ensemble(lgb_models, X_test)
    rf_test = predict_ensemble(rf_models, X_test)
    xgb_test = predict_ensemble(xgb_models, X_test)

    X_test_meta = np.column_stack([
        lgb_test, rf_test, xgb_test,
        *[X_test[feat].values for feat in meta_features]
    ])
    final_pred = meta_model.predict(X_test_meta)

    rmse = mean_squared_error(y_test, final_pred, squared=False)
    r2 = r2_score(y_test, final_pred)
    results.append((city, rmse, r2))

results_df = pd.DataFrame(results, columns=["City", "RMSE", "R2"]).sort_values("R2", ascending=False)
print("\nFinal Per-City MLP Ensemble Results:")
print(results_df)
