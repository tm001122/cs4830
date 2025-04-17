import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the line percentages and ratings CSV files
line_percentages_df = pd.read_csv('line_percentages_per_episode.csv')
ratings_df = pd.read_csv('ratings.csv')

# Merge the two datasets on Episode_Number
merged_df = pd.merge(line_percentages_df, ratings_df, on='Episode_Number')
# Export the merged dataset to a new CSV file
merged_df.to_csv('merged_dataset.csv', index=False)

# Filter for a specific character (e.g., Sheldon)
character = 'Howard'  # Change this to the character you want to analyze
character_df = merged_df[merged_df['person_scene'] == character]

# Select features (character's percentage) and target (ratings)
X = character_df[['Line_Percentage']]  # Feature: percentage of lines spoken by the character
y = character_df['Rating']  # Target: episode rating

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Display the model's coefficients
print(f"Coefficient for {character}: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

