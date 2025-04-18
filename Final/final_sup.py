import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

# Step 4: Train-Test Split for KNN
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.2, random_state=42)

# Step 5: Train K-Nearest Neighbors (KNN) Classifier
knn = KNeighborsClassifier(n_neighbors=3)  # Use 3 neighbors as an example
knn.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = knn.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Visualize KNN Predictions (Scatterplot with Cluster Descriptions in Legend)
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(
    x=X_scaled[:, 0],  # Use the first feature (e.g., Leonard)
    y=X_scaled[:, 1],  # Use the second feature (e.g., Sheldon)
    hue=df['Cluster'],  # Color by K-Means cluster
    palette='viridis',
    s=100,
    alpha=0.7
)

# Add cluster descriptions to the legend
cluster_descriptions = [
    "Cluster 0: Episodes dominated by Sheldon and Leonard.",
    "Cluster 1: Episodes where Leonard takes the lead.",
    "Cluster 2: Episodes with a focus on Penny.",
    "Cluster 3: Episodes with more contributions from Raj and Howard."
]

# Update legend labels with descriptions
handles, labels = scatter.get_legend_handles_labels()
new_labels = [f"{label}: {cluster_descriptions[int(label)]}" for label in labels if label.isdigit()]
scatter.legend(handles=handles, labels=new_labels, title="Cluster Descriptions", loc='upper right')

plt.title('KNN Predictions (Colored by K-Means Clusters)')
plt.xlabel('Leonard Line Percentage (Standardized)')
plt.ylabel('Sheldon Line Percentage (Standardized)')
plt.grid(True)
plt.savefig('knn_predictions_scatterplot_with_legend_descriptions.png')
plt.show()