# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional: to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# %%
# Step 1: Load or create a sample customer dataset
# For demonstration, here's a mock dataset
data = {
    'CustomerID': range(1, 21),
    'Recency': [30, 45, 60, 22, 80, 5, 14, 66, 120, 3, 15, 45, 70, 90, 200, 34, 55, 2, 1, 88],
    'Frequency': [5, 3, 2, 7, 1, 12, 10, 2, 1, 13, 8, 4, 2, 1, 1, 6, 5, 14, 15, 1],
    'Monetary': [500, 300, 200, 700, 100, 1200, 1000, 220, 150, 1400, 800, 400, 250, 120, 100, 650, 550, 1300, 1350, 130]
}

df = pd.DataFrame(data)
df.head()


# %%
# Step 2: Feature selection and preprocessing
features = df[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# %%
# Step 3: Elbow method to find optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot WCSS to find the "elbow"
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method - Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# %%
# Step 4: Apply KMeans with optimal k (let's assume it's 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)
df.head()


# %%
# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Cluster', palette='tab10', s=100)
plt.title('Customer Segments by Recency and Monetary Value')
plt.xlabel('Recency (days)')
plt.ylabel('Monetary Value ($)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# %%
# Optional: Analyze cluster characteristics
cluster_summary = df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
print("Cluster Summary:\n")




