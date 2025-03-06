import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load Dataset (Replace with your dataset path)
df = pd.read_csv("../data/nsl_kdd.csv")

# Drop non-numeric columns if any
df = df.select_dtypes(include=[np.number])

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# ------------------------- K-Means Clustering -------------------------
print("Running K-Means Clustering...")
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# Evaluate Clustering Performance
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# ------------------------- Autoencoder for Anomaly Detection -------------------------
print("Training Autoencoder...")

# Build Autoencoder Model
input_dim = X.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train Autoencoder
autoencoder.fit(X, X, epochs=10, batch_size=32, shuffle=True, validation_split=0.1, verbose=1)

# Compute Reconstruction Error
reconstructed = autoencoder.predict(X)
mse = np.mean(np.power(X - reconstructed, 2), axis=1)

# Plot Error Distribution
plt.figure(figsize=(8, 5))
sns.histplot(mse, bins=50, kde=True)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Mean Squared Error")
plt.ylabel("Frequency")
plt.show()

# Set threshold for anomaly detection
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

print(f"Detected {np.sum(anomalies)} anomalies out of {len(mse)} samples.")

print("Intrusion Detection Completed!")