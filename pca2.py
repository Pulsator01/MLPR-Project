import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the processed data
data = pd.read_csv("processed_data.csv")
print("Columns in data:", data.columns.tolist())

# For PCA, drop non-numeric columns ('City') and the target so that PCA runs on numeric predictors only.
features = data.drop(columns=['Invoice Quantity', 'City'])
target = data['Invoice Quantity']

# Split the data into an 80% training set and a 20% test set.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)

# ---------------------------
# Process the Training Data (80%)
# ---------------------------
# Standardize training data
scaler_train = StandardScaler()
X_train_scaled = scaler_train.fit_transform(X_train)

# Apply PCA on training data (retain enough components to explain 95% of the variance)
pca_train = PCA(n_components=0.95)
X_train_pca = pca_train.fit_transform(X_train_scaled)
print("Training PCA: Number of components =", X_train_pca.shape[1])

# Create a DataFrame for the training PCA data and add the target column.
pca_train_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
pca_train_df['Invoice Quantity'] = y_train.values

# Save the training PCA data to CSV.
pca_train_df.to_csv("pca_train.csv", index=False)

# ---------------------------
# Process the Test Data (20%)
# ---------------------------
# Standardize test data (using its own scaler, since we're processing independently)
scaler_test = StandardScaler()
X_test_scaled = scaler_test.fit_transform(X_test)

# Apply PCA on test data independently.
pca_test = PCA(n_components=0.95)
X_test_pca = pca_test.fit_transform(X_test_scaled)
print("Test PCA: Number of components =", X_test_pca.shape[1])

# Create a DataFrame for the test PCA data and add the target column.
pca_test_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])
pca_test_df['Invoice Quantity'] = y_test.values

# Save the test PCA data to CSV.
pca_test_df.to_csv("pca_test.csv", index=False)

print("PCA training data saved to 'pca_train.csv' and test data saved to 'pca_test.csv'")
