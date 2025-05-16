import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

TIMESTEPS = 30

df = pd.read_csv("/kaggle/input/data-final/data_final.csv", parse_dates=["Invoice Date"])
df = df.sort_values("Invoice Date")

df["Day"] = df["Invoice Date"].dt.day
df["Month"] = df["Invoice Date"].dt.month
df["Weekday"] = df["Invoice Date"].dt.weekday
df["7Day_MA_Sales"] = df["Invoice Quantity"].rolling(window=7).mean()
df["30Day_MA_Sales"] = df["Invoice Quantity"].rolling(window=30).mean()
df["Prev_Sales"] = df["Invoice Quantity"].shift(1)

df = df.dropna().reset_index(drop=True)

features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'precipitation', 'pressure_msl', 'wind_speed_10m',
    'Day', 'Month', 'Weekday',
    '7Day_MA_Sales', '30Day_MA_Sales', 'Prev_Sales'
]
target = 'Invoice Quantity'

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + [target]])
df_scaled = pd.DataFrame(scaled, columns=features + [target])

def create_sequences(data, target_col, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data.iloc[i:i+timesteps, :-1].values)
        y.append(data.iloc[i+timesteps][target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled, target_col=target, timesteps=TIMESTEPS)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
with open("minmax_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(" Data ready. Shapes:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
