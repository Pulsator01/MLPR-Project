import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load the data
df = pd.read_csv('finalProcessing/data_final.csv')

# Select only weather features
weather_features = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation',
    'pressure_msl', 'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'wind_direction_10m'
]
X_weather = df[weather_features]

# Standardize the weather features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_weather)

# Apply PCA to retain 95% of explained variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f'Number of PCA components to reach 95% explained variance: {X_pca.shape[1]}')
print('Explained variance ratio (weightage of each PC):', pca.explained_variance_ratio_)
print('Cumulative explained variance:', np.cumsum(pca.explained_variance_ratio_))

for i, var in enumerate(pca.explained_variance_ratio_):
    print(f'PC{i+1}: Explained Variance = {var:.4f}')

# Show the features that contribute most to each principal component
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=weather_features)
for i in range(X_pca.shape[1]):
    print(f'\nTop features for PC{i+1} (weightage: {pca.explained_variance_ratio_[i]:.4f}):')
    print(loadings.iloc[:, i].abs().sort_values(ascending=False).head(3))
