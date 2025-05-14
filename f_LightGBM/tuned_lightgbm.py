# tuned_lightgbm.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

# features
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("R2:", r2_score(y_test, pred))
