import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

city_mapping = {
    "BHP": "Bhopal", "DGP": "Durgapur", "DLI": "New Delhi", "GRN": "Gurgaon",
    "GZB": "Ghaziabad", "HBI": "Hubli", "JMU": "Jammu", "JPR": "Jaipur",
    "KCI": "Kochi", "LKW": "Lucknow", "MDI": "Madurai", "PNE": "Pune",
    "PTA": "Patna"
}

def preprocess_sales_weather(sales_file, weather_folder):
    sales_data = pd.read_excel(sales_file, sheet_name=0)
    sales_data = sales_data[['Invoice Date', 'Invoice Quantity', 'Sales Location']]
    sales_data['Invoice Date'] = pd.to_datetime(sales_data['Invoice Date'])
    sales_data['City'] = sales_data['Sales Location'].map(city_mapping)

    processed_data = []

    for city_code, city_name in city_mapping.items():
        city_sales = sales_data[sales_data['City'] == city_name]
        weather_file = os.path.join(weather_folder, f"{city_name}.csv")

        if not os.path.exists(weather_file):
            print(f"Warning: Weather data for {city_name} not found. Skipping...")
            continue

        weather_data = pd.read_csv(weather_file)
        weather_data = weather_data.loc[:, ~weather_data.columns.str.contains("^Unnamed")]
        weather_data['date'] = pd.to_datetime(weather_data['date']).dt.tz_localize(None).dt.normalize()

        city_sales = city_sales.copy()
        city_sales['Invoice Date'] = city_sales['Invoice Date'].dt.normalize()

        daily_weather = weather_data.groupby('date').mean().reset_index()
        daily_weather.rename(columns={'date': 'Invoice Date'}, inplace=True)

        merged_city_data = pd.merge(city_sales, daily_weather, on='Invoice Date', how='left')
        merged_city_data = merged_city_data.dropna(subset=['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m'])

        merged_city_data['Day'] = merged_city_data['Invoice Date'].dt.day
        merged_city_data['Month'] = merged_city_data['Invoice Date'].dt.month
        merged_city_data['Weekday'] = merged_city_data['Invoice Date'].dt.weekday

        weather_features = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
        for feature in weather_features:
            merged_city_data[f'{feature}_7day_avg'] = merged_city_data[feature].rolling(window=7, min_periods=1).mean()

        merged_city_data['City'] = city_name
        merged_city_data.drop(columns=['Invoice Date', 'Sales Location'], inplace=True)

        processed_data.append(merged_city_data)

    final_data = pd.concat(processed_data, ignore_index=True)
    final_data.reset_index(drop=True, inplace=True)
    return final_data

sales_file = "Book1.xlsx"
weather_folder = "weather_data"

preprocessed_data = preprocess_sales_weather(sales_file, weather_folder)

def analyze_features(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.isna().any().any():
        numeric_data = numeric_data.fillna(numeric_data.mean())
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(14, 12))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f', 
                cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.close()
    return corr_matrix

def apply_pca(data, n_components=None, variance_threshold=0.95):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.isna().any().any():
        numeric_data = numeric_data.dropna()
        if len(numeric_data) < len(data) * 0.7:
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            numeric_data = numeric_data.fillna(numeric_data.mean())

    if 'Volume' in numeric_data.columns:
        X = numeric_data.drop(columns=['Volume'])
        y = numeric_data['Volume']
    else:
        X = numeric_data
        y = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components is None:
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
        n_components = sum(cumulative_variance <= variance_threshold) + 1
        n_components = min(n_components, X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    column_names = [f'PC{i+1}' for i in range(n_components)]
    pca_data = pd.DataFrame(X_pca, columns=column_names)

    if y is not None:
        pca_data['Volume'] = y.values

    pca_data['City'] = preprocessed_data['City'].values

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_explained_variance.png')
    plt.close()

    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=30)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('First Two Principal Components')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pca_scatter.png')
        plt.close()

    return pca_data, pca

def interpret_pca_components(pca_model, feature_names):
    loadings = pca_model.components_
    loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=feature_names)
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.25)))
    sns.heatmap(loadings_df, cmap='coolwarm', center=0, annot=False)
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    plt.savefig('pca_loadings_heatmap.png')
    plt.close()
    return loadings_df

def visualize_clusters(pca_data):
    plt.figure(figsize=(12, 10))
    if 'Volume' in pca_data.columns:
        scatter = plt.scatter(pca_data['PC1'], pca_data['PC2'], 
                   c=pca_data['Volume'], alpha=0.6, s=30, cmap='viridis')
        plt.colorbar(scatter, label='Volume')
    else:
        plt.scatter(pca_data['PC1'], pca_data['PC2'], alpha=0.6, s=30)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: Potential Clusters in the Data')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_potential_clusters.png')
    plt.close()

corr_matrix = analyze_features(preprocessed_data)
pca_data, pca_model = apply_pca(preprocessed_data)

numeric_features = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Volume' in numeric_features:
    numeric_features.remove('Volume')

interpret_pca_components(pca_model, numeric_features)
visualize_clusters(pca_data)

preprocessed_data.reset_index(drop=True, inplace=True)
preprocessed_data = preprocessed_data.loc[:, ~preprocessed_data.columns.str.contains("^Unnamed")]
preprocessed_data.to_csv("processed_data.csv", index=False)

pca_data = pca_data.loc[:, ~pca_data.columns.str.contains("^Unnamed")]
pca_data.to_csv("pca_data.csv", index=False)
