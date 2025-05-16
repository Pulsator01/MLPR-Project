import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

with open("minmax_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

target_index = scaler.feature_names_in_.tolist().index('Invoice Quantity')
n_features = len(scaler.feature_names_in_)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
    verbose=1
)

y_pred = model.predict(X_test)

y_test_full = np.zeros((len(y_test), n_features))
y_test_full[:, target_index] = y_test
y_test_inv = scaler.inverse_transform(y_test_full)[:, target_index]

y_pred_full = np.zeros((len(y_pred), n_features))
y_pred_full[:, target_index] = y_pred.ravel()
y_pred_inv = scaler.inverse_transform(y_pred_full)[:, target_index]

rmse = mean_squared_error(y_test_inv, y_pred_inv, squared=False)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f" Final RMSE: {rmse:.2f}")
print(f" Final R²: {r2:.3f}")
