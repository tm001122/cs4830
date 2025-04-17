import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Add a target column to label the data
data['target'] = data['Unnamed: 0'].apply(lambda x: 'brca' if 'brca' in x else ('prad' if 'prad' in x else ('luad' if 'luad' in x else 'unknown')))

# Filter to only have breast cancer, prad cancer, or luad cancer data
data = data[data['target'].isin(['brca', 'prad', 'luad'])]

# Map cancer types to numeric labels
label_mapping = {'brca': 0, 'prad': 1, 'luad': 2}
data['true_labels'] = data['target'].map(label_mapping)

# Drop non-numeric columns
data = data.select_dtypes(include=[float, int])

# Handle missing values if any
data = data.dropna()

# Assuming the dataset has features in columns and no labels
X = data.drop(columns=['true_labels']).values
true_labels = data['true_labels'].values

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Dimensionality reduction using t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Dimensionality reduction using UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X)

# Visualize the results of PCA, t-SNE, and UMAP
methods = [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]
colors = ['blue', 'green', 'red']  # Colors for each method

# Individual graphs for each method
for method_name, X_reduced in methods:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=true_labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f'{method_name} Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='True Labels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_visualization.png')
    plt.show()

# Combined graph for PCA, t-SNE, and UMAP
plt.figure(figsize=(10, 8))
for (method_name, X_reduced), color in zip(methods, colors):
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], label=method_name, alpha=0.6, s=50, color=color)

plt.title('Combined Visualization of PCA, t-SNE, and UMAP')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('combined_visualization.png')
plt.show()

# Perform clustering on the reduced data (e.g., PCA)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Evaluate clustering
silhouette_avg = silhouette_score(X_pca, labels)
ch_score = calinski_harabasz_score(X_pca, labels)
dbi_score = davies_bouldin_score(X_pca, labels)
ri_score = rand_score(true_labels, labels)
ari_score = adjusted_rand_score(true_labels, labels)
mi_score = mutual_info_score(true_labels, labels)
nmi_score = normalized_mutual_info_score(true_labels, labels)

# Print the evaluation metrics
print('Clustering Evaluation Metrics (on PCA-reduced data):')
print(f'Silhouette Score: {silhouette_avg}')
print(f'Calinski-Harabasz Score: {ch_score}')
print(f'Davies-Bouldin Index: {dbi_score}')
print(f'Rand Index: {ri_score}')
print(f'Adjusted Rand Index: {ari_score}')
print(f'Mutual Information: {mi_score}')
print(f'Normalized Mutual Information: {nmi_score}')

