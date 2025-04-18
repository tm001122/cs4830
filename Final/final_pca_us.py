import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('tbbt_character_analysis.csv')

# Step 2: Preprocess the data
# Exclude non-numeric columns (e.g., title, season, episode, rating)
character_columns = ['Leonard', 'Sheldon', 'Penny', 'Raj', 'Howard', 'Bernadette', 'Amy', 'Stuart']
X = df[character_columns]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Rating'] = df['rating']

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Choose 3 clusters as an example
pca_df['Cluster'] = kmeans.fit_predict(X_pca)

# Step 5: Visualize the Results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100)
plt.title('K-Means Clustering on PCA-Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('kmeans_pca_clusters.png')

# Optional: Analyze the clusters
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers (in PCA space):")
print(cluster_centers)

# Step 6: Visualize how ratings are distributed across clusters
plt.figure(figsize=(10, 6))
sns.boxplot(data=pca_df, x='Cluster', y='Rating', palette='viridis')
plt.title('Episode Ratings by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Rating')
plt.grid(True)
plt.show()