# Load the dataset
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import SelectKBest, f_classif

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

# Lists to store the scores for plotting
k_values = []
silhouette_scores = []
ch_scores = []
dbi_scores = []
ri_scores = []
ari_scores = []
mi_scores = []
nmi_scores = []

# Open a CSV file to write the evaluation metrics
with open('3-19.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['k', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Index', 'Rand Index', 'Adjusted Rand Index', 'Mutual Information', 'Normalized Mutual Information'])

    # Perform K-means clustering for different values of k
    for k in range(1, 1000, 100):
        # Select the top k features
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, true_labels)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_new)

        # Evaluate the clusters
        silhouette_avg = silhouette_score(X_new, labels)
        ch_score = calinski_harabasz_score(X_new, labels)
        dbi_score = davies_bouldin_score(X_new, labels)
        ri_score = rand_score(true_labels, labels)
        ari_score = adjusted_rand_score(true_labels, labels)
        mi_score = mutual_info_score(true_labels, labels)
        nmi_score = normalized_mutual_info_score(true_labels, labels)

        # Write the evaluation metrics to the CSV file
        writer.writerow([k, silhouette_avg, ch_score, dbi_score, ri_score, ari_score, mi_score, nmi_score])

        # Store the scores for plotting
        k_values.append(k)
        silhouette_scores.append(silhouette_avg)
        ch_scores.append(ch_score)
        dbi_scores.append(dbi_score)
        ri_scores.append(ri_score)
        ari_scores.append(ari_score)
        mi_scores.append(mi_score)
        nmi_scores.append(nmi_score)

        # Print the evaluation metrics
        print(f'For k={k}:')
        print(f'Silhouette Score: {silhouette_avg}')
        print(f'Calinski-Harabasz Score: {ch_score}')
        print(f'Davies-Bouldin Index: {dbi_score}')
        print(f'Rand Index: {ri_score}')
        print(f'Adjusted Rand Index: {ari_score}')
        print(f'Mutual Information: {mi_score}')
        print(f'Normalized Mutual Information: {nmi_score}')
        print('-----------------------------------')

# Plot and save the scores for each metric
metrics = [
    ('Silhouette Score', silhouette_scores),
    ('Calinski-Harabasz Score', ch_scores),
    ('Davies-Bouldin Index', dbi_scores),
    ('Rand Index', ri_scores),
    ('Adjusted Rand Index', ari_scores),
    ('Mutual Information', mi_scores),
    ('Normalized Mutual Information', nmi_scores)
]

for metric_name, metric_scores in metrics:
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, metric_scores, label=metric_name)
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title(f'{metric_name} for Different k Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name.replace(" ", "_").lower()}.png')
    plt.close()

