import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load datasets
sales_data = pd.read_excel('Book1.xlsx')
weather_data = pd.read_csv('New Delhi.csv')

# Convert date columns to datetime and ensure consistent timezone handling
sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])
weather_data['date'] = pd.to_datetime(weather_data['date']).dt.tz_localize(None)  # Remove timezone info

# Merge datasets
merged_data = pd.merge(sales_data, weather_data, left_on='Invoice Date', right_on='date', how='left')

# Feature engineering
merged_data['day_of_week'] = merged_data['Invoice Date'].dt.dayofweek
merged_data['month'] = merged_data['Invoice Date'].dt.month
merged_data['is_weekend'] = merged_data['day_of_week'].isin([5, 6]).astype(int)

# Select relevant features
features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature', 
            'precipitation', 'cloud_cover', 'day_of_week', 'month', 'is_weekend']

# Handle missing values
merged_data[features] = merged_data[features].interpolate()

# Normalize features
scaler = MinMaxScaler()
merged_data[features] = scaler.fit_transform(merged_data[features])

# Create lagged features
merged_data['sales_lag_1'] = merged_data.groupby('Sales Location')['Invoice Quantity'].shift(1)
merged_data['temp_lag_1'] = merged_data.groupby('Sales Location')['temperature_2m'].shift(1)

# Final dataset
final_data = merged_data.dropna().reset_index(drop=True)

# Train-test split (example)
train_data = final_data[final_data['Invoice Date'] < '2024-01-01']
test_data = final_data[final_data['Invoice Date'] >= '2024-01-01']

# Save processed data to CSV files
train_data.to_csv('processed_train_data.csv', index=False)
test_data.to_csv('processed_test_data.csv', index=False)
