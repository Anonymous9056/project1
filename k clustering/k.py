# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Mall_Customers.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Select relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method for Optimal K (no colors for individual points)
wcss = []  # Within-cluster sum of squares
for k in range(1, 11):  # Trying different numbers of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b', markersize=10)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# Define the optimal number of clusters based on the elbow method
optimal_clusters = 5

# Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# Plot each cluster with specified colors
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_clusters):
    plt.scatter(
        scaled_features[cluster_labels == i, 0],
        scaled_features[cluster_labels == i, 1],
        s=100,
        c=colors[i],
        label=f'Cluster {i + 1}'
    )

# Plot the centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='yellow',
    marker='X',
    label='Centroids'
)

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$) (Standardized)')
plt.ylabel('Spending Score (1-100) (Standardized)')
plt.legend()
plt.grid()
plt.show()
