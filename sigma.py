import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Mapping of sales location codes to city names
city_mapping = {
    "BHP": "Bhopal", "DGP": "Durgapur", "DLI": "New Delhi", "GRN": "Gurgaon",
    "GZB": "Ghaziabad", "HBI": "Hubli", "JMU": "Jammu", "JPR": "Jaipur",
    "KCI": "Kochi", "LKW": "Lucknow", "MDI": "Madurai", "PNE": "Pune",
    "PTA": "Patna"
}

def preprocess_sales_weather(sales_file, weather_folder):
    # Load the sales data
    sales_data = pd.read_excel(sales_file, sheet_name=0)
    sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])

    # Replace sales location codes with city names
    sales_data['City'] = sales_data['Sales Location'].map(city_mapping)

    # Process each city separately
    processed_data = []

    for city_code, city_name in city_mapping.items():
        city_sales = sales_data[sales_data['City'] == city_name]

        # Construct weather file path
        weather_file = os.path.join(weather_folder, f"{city_name}.csv")
        
        if not os.path.exists(weather_file):
            print(f"Warning: Weather data for {city_name} not found. Skipping...")
            continue

        # Load weather data
        weather_data = pd.read_csv(weather_file)
        weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

        # Aggregate weather data to daily level
        daily_weather = weather_data.groupby('date').mean().reset_index()
        daily_weather.rename(columns={'date': 'Invoice Date'}, inplace=True)

        # Merge sales and weather data
        daily_weather['Invoice Date'] = pd.to_datetime(daily_weather['Invoice Date'])
        merged_city_data = pd.merge(city_sales, daily_weather, on='Invoice Date', how='left')

        # Handle missing values
        merged_city_data = merged_city_data.ffill()

        # Feature Engineering
        merged_city_data['Day'] = merged_city_data['Invoice Date'].dt.day
        merged_city_data['Month'] = merged_city_data['Invoice Date'].dt.month
        merged_city_data['Weekday'] = merged_city_data['Invoice Date'].dt.weekday

        # Compute 7-day rolling averages for weather variables
        weather_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
        for feature in weather_features:
            merged_city_data[f'{feature}_7day_avg'] = merged_city_data[feature].rolling(window=7, min_periods=1).mean()

        # Encode categorical variables using one-hot encoding
        merged_city_data = pd.get_dummies(merged_city_data, columns=['Sales Zone', 'Sales Channel'], drop_first=True)

        # Drop unnecessary columns
        merged_city_data.drop(columns=['Invoice Date', 'Sales Location', 'City'], inplace=True)

        processed_data.append(merged_city_data)

    # Concatenate processed data for all cities
    final_data = pd.concat(processed_data, ignore_index=True)

    return final_data

# Example Usage
sales_file = "Book1.xlsx"  # Path to your sales data file
weather_folder = "weather_data"  # Folder containing weather CSV files for all cities

preprocessed_data = preprocess_sales_weather(sales_file, weather_folder)

# Save the processed data to a CSV file
preprocessed_data.to_csv("processed_data.csv", index=False)

# Display the processed data
print("Processed data shape:", preprocessed_data.shape)
print("First few rows of processed data:")
print(preprocessed_data.head())
