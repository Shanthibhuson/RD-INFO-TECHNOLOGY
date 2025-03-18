import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulated Network Traffic Data with different values
data = {
    'src_ip': ['192.168.2.1', '192.168.2.2', '192.168.2.3', '10.1.0.1', '10.1.0.2', '10.1.0.3', '192.168.2.4'],
    'dst_ip': ['192.168.2.5', '192.168.2.6', '192.168.2.7', '10.1.0.4', '10.1.0.5', '10.1.0.6', '192.168.2.8'],
    'protocol': [1, 3, 2, 1, 2, 3, 1],  # 1 = TCP, 2 = UDP, 3 = ICMP
    'bytes_sent': [500, 2500, 750, 9000, 300, 1500, 11000],
    'bytes_received': [1500, 750, 1500, 800, 1400, 700, 5000],
    'connection_duration': [50, 110, 25, 200, 55, 80, 320]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Feature selection
features = df[['protocol', 'bytes_sent', 'bytes_received', 'connection_duration']]

# Standardize the feature values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(features_scaled)

df['cluster'] = kmeans.predict(features_scaled)

# Compute distance to centroid
df['distance_to_centroid'] = np.linalg.norm(features_scaled - kmeans.cluster_centers_[df['cluster']], axis=1)

# Set anomaly threshold (top 5% as anomalies)
threshold = np.percentile(df['distance_to_centroid'], 95)
df['anomaly'] = df['distance_to_centroid'] > threshold

# Display results
print(df)

# Visualization
plt.scatter(df['bytes_sent'], df['bytes_received'], c=df['anomaly'], cmap='coolwarm', marker='o')
plt.title('Network Traffic Anomalies (Red = Anomalies)')
plt.xlabel('Bytes Sent')
plt.ylabel('Bytes Received')
plt.show()
