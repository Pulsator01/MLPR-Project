# baseline_lightgbm.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday
df = df.dropna()

X = df.drop(columns=["Invoice Date", "Invoice Quantity", "Sales Zone", "Sales Location"])
y = df["Invoice Quantity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluation
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("R2:", r2_score(y_test, pred))
