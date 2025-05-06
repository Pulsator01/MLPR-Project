import pandas as pd

# Mapping from city codes to city names
city_mapping = {
    "BHP": "Bhopal", "DGP": "Durgapur", "DLI": "New Delhi", "GRN": "Gurgaon",
    "GZB": "Ghaziabad", "HBI": "Hubli", "JMU": "Jammu", "JPR": "Jaipur",
    "KCI": "Kochi", "LKW": "Lucknow", "MDI": "Madurai", "PNE": "Pune",
    "PTA": "Patna"
}

# Load weather data
weather_df = pd.read_csv('finalProcessing/final_weather_data.csv')
weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
weather_df['city'] = weather_df['city'].str.strip()

# Load sales data
sales_df = pd.read_csv('finalProcessing/sales_data.csv')
sales_df['Invoice Date'] = pd.to_datetime(sales_df['Invoice Date']).dt.normalize()
sales_df['Sales Location'] = sales_df['Sales Location'].map(city_mapping)
sales_df['Sales Location'] = sales_df['Sales Location'].str.strip()

# Merge on date and city
merged_df = pd.merge(
    sales_df,
    weather_df,
    left_on=['Invoice Date', 'Sales Location'],
    right_on=['date', 'city'],
    how='inner'
)

# Drop duplicate columns
merged_df = merged_df.drop(columns=['date', 'city'])

# Save the merged data
merged_df.to_csv('finalProcessing/data_final.csv', index=False)
