import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

mutation_columns = data.columns[1:]

# Remove rows with more than 6 mutations
data = data[(data[mutation_columns] <= 5).all(axis=1)]

# Add a target column to label the data
data['target'] = data['Unnamed: 0'].apply(lambda x: 'brca' if 'brca' in x else ('prad' if 'prad' in x else ('luad' if 'luad' in x else 'unknown')))

# Filter to only have breast cancer, prad cancer, or luad cancer data
br_data = data[data['Unnamed: 0'].str.contains('brca', na=False)]
prad_data = data[data['Unnamed: 0'].str.contains('prad', na=False)]
luad_data = data[data['Unnamed: 0'].str.contains('luad', na=False)]

# Sample number of rows from each type of cancer data
number_of_each = 300 # max is 493 for lung cancer
br_sample = br_data.sample(n=number_of_each, random_state=1)
prad_sample = prad_data.sample(n=number_of_each, random_state=1)
luad_sample = luad_data.sample(n=number_of_each, random_state=1)

# Reserve number from each sample set for testing
number_for_testing = 100
br_train, br_test = train_test_split(br_sample, test_size=number_for_testing, random_state=1)
prad_train, prad_test = train_test_split(prad_sample, test_size=number_for_testing, random_state=1)
luad_train, luad_test = train_test_split(luad_sample, test_size=number_for_testing, random_state=1)

# Combine the training samples into a single dataframe
combined_train_data = pd.concat([br_train, prad_train, luad_train])

# Combine the testing samples into a single dataframe
combined_test_data = pd.concat([br_test, prad_test, luad_test])

# Prepare the training data
X_train = combined_train_data[mutation_columns]
y_train = combined_train_data['target']

# Prepare the testing data
X_test = combined_test_data[mutation_columns]
y_test = combined_test_data['target']

# Initialize the perceptron model
perc = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Train the perceptron model
perc.fit(X_train, y_train)

# Make predictions
y_pred = perc.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the perceptron model: {accuracy*100}")