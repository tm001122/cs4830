import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
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

silhouette_scores = []
ch_scores = []
dbi_scores = []
ri_scores = []
ari_scores = []
mi_scores = []
nmi_scores = []

X=X.fillna(0)

X = X.loc[:, (X!= X.iloc[0]).any()]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform clustering
dbscan = DBSCAN(eps=10, min_samples=2)
clusters = dbscan.fit_predict(X)

# Evaluate clustering
silhouette_avg = silhouette_score(X_pca, clusters)
ch_score = calinski_harabasz_score(X_pca, clusters)
dbi_score = davies_bouldin_score(X_pca, clusters)
ri_score = rand_score(y, clusters)
ari_score = adjusted_rand_score(y, clusters)
mi_score = adjusted_mutual_info_score(y, clusters)
nmi_score = normalized_mutual_info_score(y, clusters)

silhouette_scores.append(silhouette_avg)
ch_scores.append(ch_score)
dbi_scores.append(dbi_score)
ri_scores.append(ri_score)
ari_scores.append(ari_score)
mi_scores.append(mi_score)
nmi_scores.append(nmi_score)
# Print the evaluation metrics
print(f'For eps=10, min_samples=2:')
print(f'Silhouette Score: {silhouette_avg}')
print(f'Calinski-Harabasz Score: {ch_score}')
print(f'Davies-Bouldin Index: {dbi_score}')
print(f'Rand Index: {ri_score}')
print(f'Adjusted Rand Index: {ari_score}')
print(f'Mutual Information: {mi_score}')
print(f'Normalized Mutual Information: {nmi_score}')
print('-----------------------------------')
# Save results to a CSV file
with open('dbscan_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['eps', 'min_samples', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index', 'Rand Index', 'Adjusted Rand Index', 'Mutual Information', 'Normalized Mutual Information'])
    # Write the results
    writer.writerow([10, 2, silhouette_avg, ch_score, dbi_score, ri_score, ari_score, mi_score, nmi_score])

    # Plot the evaluation metrics on a bar graph
    metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index', 
               'Rand Index', 'Adjusted Rand Index', 'Mutual Information', 'Normalized Mutual Information']
    values = [silhouette_avg, ch_score, dbi_score, ri_score, ari_score, mi_score, nmi_score]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=metrics, y=values, palette='viridis')
    plt.title('Clustering Evaluation Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    plt.savefig('clustering_metrics.png')