import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/data-final/data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

le_zone = LabelEncoder()
le_loc = LabelEncoder()
df["Sales Zone"] = le_zone.fit_transform(df["Sales Zone"])
df["Sales Location"] = le_loc.fit_transform(df["Sales Location"])

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday

df["Prev_Day_Sales"] = df["Invoice Quantity"].shift(1)
df["7Day_MA_Sales"] = df["Invoice Quantity"].rolling(window=7).mean()
df = df.dropna()

X = df.drop(columns=["Invoice Date", "Invoice Quantity"])
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

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f" R² Score: {r2:.2f}")

lgb.plot_importance(model, max_num_features=15)
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
