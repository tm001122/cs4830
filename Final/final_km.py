import pandas as pd
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

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Choose 4 clusters as an example
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize the Clusters as a Scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='Leonard',  # Use 'Leonard' as the x-axis
    y='Sheldon',  # Use 'Sheldon' as the y-axis
    hue='Cluster',
    palette='viridis',
    s=100
)
plt.title('K-Means Clustering (Without PCA)')
plt.xlabel('Leonard Line Percentage')
plt.ylabel('Sheldon Line Percentage')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('kmeans_scatterplot_without_pca.png')
plt.show()

# Optional: Analyze the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=character_columns)
print("Cluster Centers (in original feature space):")
print(cluster_centers)