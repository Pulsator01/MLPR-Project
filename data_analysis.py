import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the processed data
data = pd.read_csv('processed_data.csv')

# Basic information about the data
print("Dataset shape:", data.shape)
print("\nData types:")
print(data.dtypes)
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Split data based on invoice quantity
high_quantity = data[data['Invoice Quantity'] > 30]
low_quantity = data[data['Invoice Quantity'] <= 30]

print("\nHigh quantity orders (>30):", high_quantity.shape[0])
print("Low quantity orders (<=30):", low_quantity.shape[0])

# Analyze correlation between weather variables and Invoice Quantity
weather_cols = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                'apparent_temperature', 'precipitation', 'wind_speed_10m',
                'cloud_cover', 'temperature_2m_7day_avg', 
                'relative_humidity_2m_7day_avg', 'wind_speed_10m_7day_avg']

# Create correlation matrix for weather features with Invoice Quantity
corr_matrix = data[['Invoice Quantity'] + weather_cols].corr()
print("\nCorrelation with Invoice Quantity:")
print(corr_matrix['Invoice Quantity'].sort_values(ascending=False))

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Weather Variables and Invoice Quantity")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")

# Plot invoice quantity vs temperature
plt.figure(figsize=(10, 6))
plt.scatter(data['temperature_2m'], data['Invoice Quantity'], alpha=0.6)
plt.title('Invoice Quantity vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Invoice Quantity')
plt.grid(True, alpha=0.3)
plt.savefig("temp_vs_invoice.png")

# Plot invoice quantity vs humidity
plt.figure(figsize=(10, 6))
plt.scatter(data['relative_humidity_2m'], data['Invoice Quantity'], alpha=0.6)
plt.title('Invoice Quantity vs Relative Humidity')
plt.xlabel('Relative Humidity (%)')
plt.ylabel('Invoice Quantity')
plt.grid(True, alpha=0.3)
plt.savefig("humidity_vs_invoice.png")

# Monthly analysis
monthly_data = data.groupby('Month')['Invoice Quantity'].agg(['mean', 'sum', 'count'])
print("\nMonthly Invoice Analysis:")
print(monthly_data)

# Plot monthly invoice quantities
plt.figure(figsize=(12, 6))
monthly_data['sum'].plot(kind='bar', color='skyblue')
plt.title('Total Invoice Quantity by Month')
plt.xlabel('Month')
plt.ylabel('Total Invoice Quantity')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig("monthly_invoice.png")

# Simple Random Forest model to predict Invoice Quantity
print("\nTraining a Random Forest Regressor...")

# Prepare features and target
X = data[weather_cols]
y = data['Invoice Quantity']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': weather_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Predicting Invoice Quantity')
plt.tight_layout()
plt.savefig("feature_importance.png")

print("\nAnalysis complete. Plots saved to current directory.")