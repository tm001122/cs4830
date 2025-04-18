from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import numpy as np

result_df = pd.read_csv('tbbt_character_analysis.csv')

features = ['Leonard', 'Sheldon', 'Penny', 'Raj', 'Howard', 'Bernadette', 'Amy', 'Stuart']
X = result_df[features].fillna(0)  # Just in case
 
# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
loadings = pd.DataFrame(pca.components_, columns=features, index=['PC1', 'PC2'])
print(loadings.T)
 
# Plot PCA result
# Character features
features = ['Leonard', 'Sheldon', 'Penny', 'Raj', 'Howard', 'Bernadette', 'Amy', 'Stuart']
X = result_df[features].fillna(0)
 
# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=result_df['rating'], cmap='viridis')
plt.colorbar(label='IMDB Rating')
plt.title('Episodes visualized by PCA of character line percentages')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('pca_character_line_percentages.png')
 
# K-means clustering (4 types of episodes)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)
centers = kmeans.cluster_centers_
 
# Map cluster numbers to meaningful names
cluster_labels = {
    0: "The Amy & Sheldon Duo",
    1: "Balanced Classic Cast",
    2: "Sheldon & Leonard Power",
    3: "Howard & Raj Sideplots"
}
 
# Add results to dataframe
result_df['PC1'] = X_pca[:, 0]
result_df['PC2'] = X_pca[:, 1]
result_df['cluster'] = clusters
result_df['cluster_label'] = result_df['cluster'].map(cluster_labels)
 
# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    result_df['PC1'], result_df['PC2'],
    c=result_df['cluster'],
    cmap='tab10',
    edgecolor='k',
    s=80
)
plt.colorbar(scatter, label='Cluster ID')
plt.title('TBBT Episodes by Dialogue Type (Labeled Clusters)')
plt.xlabel('PC1: Leonard/Sheldon ←→ Amy/Bernadette')
plt.ylabel('PC2: Sheldon/Amy ↕ Raj/Howard')
plt.grid(True)
 
# Cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Cluster Centers')
 
# Label cluster centers with names
for i, (x, y) in enumerate(centers):
    plt.text(x, y, cluster_labels[i], fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
 
plt.legend()
plt.savefig('kmeans_pca_clusters.png')