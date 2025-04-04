import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import csv

# Load the cancer data from sklearn
data = load_breast_cancer()
target = pd.Series(data.target)
data = pd.DataFrame(data.data, columns=data.feature_names)
data.index = data.index.astype(str)
y = target
X = data

# Print the initial data
# print("Initial Data:")
# print(X.head())
# print(y.head())

# Define parameter ranges to test
n_clusters_list = [2, 3, 4, 5]
linkage_methods = ['ward', 'complete', 'average', 'single']

# Store results
results = []

# Perform hierarchical clustering for different parameters
for n_clusters in n_clusters_list:
    for linkage in linkage_methods:
        
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='euclidean')
        labels = clustering.fit_predict(X)

        # Evaluate clustering
        silhouette_avg = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        dbi_score = davies_bouldin_score(X, labels)
        ri_score = rand_score(y, labels)
        ari_score = adjusted_rand_score(y, labels)
        mi_score = adjusted_mutual_info_score(y, labels)
        nmi_score = normalized_mutual_info_score(y, labels)

        # Store results
        results.append({
            'n_clusters': n_clusters,
            'linkage': linkage,
            'Silhouette Score': silhouette_avg,
            'CH Score': ch_score,
            'DBI Score': dbi_score,
            'Rand Index': ri_score,
            'Adjusted Rand Index': ari_score,
            'Mutual Information': mi_score,
            'Normalized Mutual Information': nmi_score
        })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('clustering_results.csv', index=False)

# Plot each evaluation metric
metrics = [
    'Silhouette Score',
    'CH Score',
    'DBI Score',
    'Rand Index',
    'Adjusted Rand Index',
    'Mutual Information',
    'Normalized Mutual Information'
]

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for linkage in linkage_methods:
        subset = results_df[results_df['linkage'] == linkage]
        plt.plot(subset['n_clusters'], subset[metric], label=f'Linkage: {linkage}')
    plt.title(f'{metric} vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric.replace(" ", "_").lower()}_vs_clusters.png')
    plt.close()

print("Results saved to 'clustering_results.csv' and graphs saved as PNG files.")