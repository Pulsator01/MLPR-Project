import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    
    # Keep only the required columns from the sales data
    sales_data = sales_data[['Invoice Date', 'Invoice Quantity', 'Sales Location']]
    
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

# Add feature correlation analysis before PCA
def analyze_features(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    print(f"Analyzing {len(numeric_data.columns)} numeric features...")
    if numeric_data.isna().any().any():
        print(f"Filling {numeric_data.isna().sum().sum()} NaN values temporarily for analysis...")
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
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                       corr_matrix.iloc[i, j]))
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|correlation| > 0.7):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"{feat1} and {feat2}: {corr:.3f}")
    else:
        print("\nNo feature pairs with |correlation| > 0.7 found.")
    plt.figure(figsize=(20, 15))
    for i, column in enumerate(numeric_data.columns[:min(15, len(numeric_data.columns))]):
        plt.subplot(3, 5, i+1)
        sns.histplot(numeric_data[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    return corr_matrix


# Apply PCA analysis
def apply_pca(data, n_components=None, variance_threshold=0.95):
    """
    Apply PCA to the preprocessed data.
    
    Args:
        data: Preprocessed DataFrame
        n_components: Number of PCA components (if None, will use variance_threshold)
        variance_threshold: Amount of variance to retain (default 0.95)
    
    Returns:
        pca_data: DataFrame with PCA components
        pca: Fitted PCA model
    """
    # Drop any categorical columns or columns with non-numeric data
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Check for and handle NaN values
    if numeric_data.isna().any().any():
        print(f"Warning: Found {numeric_data.isna().sum().sum()} NaN values in the data.")
        print("Removing rows with NaN values...")
        numeric_data = numeric_data.dropna()
        print(f"Remaining rows after NaN removal: {len(numeric_data)}")
        
        # If too many rows were removed, we could instead use imputation
        if len(numeric_data) < len(data) * 0.7:  # If we lost more than 30% of data
            print("Too many rows lost. Using mean imputation instead...")
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            # Fill NaN values with mean of each column
            numeric_data = numeric_data.fillna(numeric_data.mean())
    
    # Remove the target variable if it exists
    if 'Volume' in numeric_data.columns:
        X = numeric_data.drop(columns=['Volume'])
        y = numeric_data['Volume']
    else:
        X = numeric_data
        y = None
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of components based on variance threshold
    if n_components is None:
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
        n_components = sum(cumulative_variance <= variance_threshold) + 1
        # Ensure n_components doesn't exceed the number of features
        n_components = min(n_components, X_scaled.shape[1])
    
    # Apply PCA with determined number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA components
    column_names = [f'PC{i+1}' for i in range(n_components)]
    pca_data = pd.DataFrame(X_pca, columns=column_names)
    
    # Add back the target variable if it exists
    if y is not None:
        pca_data['Volume'] = y.values
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Components')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    
    # Plot first two principal components
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=30)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('First Two Principal Components')
        plt.grid(True)
        plt.savefig('pca_scatter.png')
    
    print(f"PCA applied with {n_components} components, retaining {pca.explained_variance_ratio_.sum():.2%} of variance")
    
    return pca_data, pca

# Add PCA interpretation functions
def interpret_pca_components(pca_model, feature_names):
    """
    Create visualizations and analysis to interpret PCA components.
    
    Args:
        pca_model: Fitted PCA model
        feature_names: List of original feature names
    """
    # Get component loadings (weights)
    loadings = pca_model.components_
    
    # Create a DataFrame of loadings
    loadings_df = pd.DataFrame(
        loadings.T, 
        columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
        index=feature_names
    )
    
    # Plot loadings heatmap for visualization
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.25)))
    sns.heatmap(loadings_df, cmap='coolwarm', center=0, annot=False)
    plt.title('PCA Component Loadings')
    plt.tight_layout()
    plt.savefig('pca_loadings_heatmap.png')
    
    # Plot feature importance for top components
    num_components_to_plot = min(3, loadings.shape[0])
    for i in range(num_components_to_plot):
        plt.figure(figsize=(12, 8))
        component = loadings[i]
        component_df = pd.DataFrame({
            'Feature': feature_names,
            'Loading': component
        })
        component_df = component_df.reindex(component_df['Loading'].abs().sort_values(ascending=False).index)
        
        sns.barplot(x='Loading', y='Feature', data=component_df.head(15), palette='viridis')
        plt.title(f'Top 15 Features in Principal Component {i+1}')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'pca_pc{i+1}_top_features.png')
    
    return loadings_df

# Add clustering visualization based on PCA
def visualize_clusters(pca_data):
    """
    Create a visualization that may reveal natural clusters in the PCA-transformed data.
    
    Args:
        pca_data: DataFrame with PCA components
    """
    # Use just the first two components for visualization
    plt.figure(figsize=(12, 10))
    
    # If 'Volume' column exists, use it for coloring points
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
    
    # Add density contours to highlight clusters
    try:
        from scipy.stats import gaussian_kde
        # Calculate the point density
        xy = np.vstack([pca_data['PC1'], pca_data['PC2']])
        z = gaussian_kde(xy)(xy)
        
        # Sort the points by density
        idx = z.argsort()
        x, y, z = pca_data['PC1'].iloc[idx], pca_data['PC2'].iloc[idx], z[idx]
        
        plt.figure(figsize=(12, 10))
        plt.scatter(x, y, c=z, s=30, alpha=0.6, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA: Data Density Visualization')
        plt.grid(True, alpha=0.3)
        plt.savefig('pca_density_clusters.png')
    except:
        print("Could not create density visualization. Skipping...")
    
    plt.savefig('pca_potential_clusters.png')

# Run the additional analyses
print("\n--- Performing Feature Correlation Analysis ---")
corr_matrix = analyze_features(preprocessed_data)

# Apply PCA to the preprocessed data
pca_data, pca_model = apply_pca(preprocessed_data)

# Get feature names for PCA interpretation
numeric_features = preprocessed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'Volume' in numeric_features:
    numeric_features.remove('Volume')  # Remove target if present

print("\n--- Interpreting PCA Components ---")
loadings_df = interpret_pca_components(pca_model, numeric_features)

print("\n--- Visualizing Potential Clusters ---")
visualize_clusters(pca_data)

# Save both the original processed data and PCA-transformed data
preprocessed_data.to_csv("processed_data.csv", index=False)
pca_data.to_csv("pca_data.csv", index=False)

# Add analysis of feature importance to PCA components
print("\n--- Most Important Features for Top 3 PCA Components ---")
for i in range(min(3, len(pca_model.components_))):
    component = pca_model.components_[i]
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': np.abs(component)
    }).sort_values('Importance', ascending=False)
    
    print(f"\nPC{i+1} explains {pca_model.explained_variance_ratio_[i]:.2%} of variance")
    print("Top 5 features:")
    for idx, row in feature_importance.head(5).iterrows():
        direction = "+" if pca_model.components_[i][idx] > 0 else "-"
        print(f"  {row['Feature']} ({direction}): {row['Importance']:.4f}")

# Display the processed data
print("\nProcessed data shape:", preprocessed_data.shape)
print("First few rows of processed data:")
print(preprocessed_data.head())

# Display PCA data
print("\nPCA data shape:", pca_data.shape) 
print("First few rows of PCA data:")
print(pca_data.head())

# Display PCA components importance
print("\nExplained variance ratio by component:")
cumulative = 0
for i, ratio in enumerate(pca_model.explained_variance_ratio_):
    cumulative += ratio
    print(f"PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%) - Cumulative: {cumulative:.4f} ({cumulative*100:.2f}%)")

# Add a visualization of cumulative explained variance
plt.figure(figsize=(12, 6))
cumulative_variance = pca_model.explained_variance_ratio_.cumsum()
plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
        pca_model.explained_variance_ratio_, 
        alpha=0.6, 
        label='Individual Explained Variance')
plt.step(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 
         where='mid', 
         label='Cumulative Explained Variance', 
         color='red')
plt.axhline(y=0.95, color='k', linestyle='--', label='95% Threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('pca_variance_components.png')
