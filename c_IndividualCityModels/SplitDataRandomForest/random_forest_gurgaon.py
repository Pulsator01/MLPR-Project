import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


df = pd.read_csv('finalProcessing/data_final.csv')
df_gurgaon = df[df['Sales Location'] == 'Gurgaon']

weather_features = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'cloud_cover', 'wind_speed_10m'
]

# If there are any categorical columns left, get dummies (shouldn't be needed for only Gurgaon)
# But if you want to keep 'Sales Zone', you can one-hot encode it
if 'Sales Zone' in df_gurgaon.columns:
    df_gurgaon = pd.get_dummies(df_gurgaon, columns=['Sales Zone'], drop_first=True)

X = df_gurgaon[weather_features + [col for col in df_gurgaon.columns if col.startswith('Sales Zone_')]]
y = df_gurgaon['Invoice Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R^2 Score: {r2:.4f}')
