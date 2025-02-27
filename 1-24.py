import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Display the initial dataset
print("Initial dataset:")
print(data)

# Filter to only have breast cancer data
for index, row in data.iterrows():
    if 'brca' in str(row['Unnamed: 0']):
        data.at[index, 'is_breast_cancer'] = True
    else:
        data.at[index, 'is_breast_cancer'] = False

for index, row in data.iterrows():
    if 'prad' in str(row['Unnamed: 0']):
        data.at[index, 'is_prad_cancer'] = True
    else:
        data.at[index, 'is_prad_cancer'] = False

for index, row in data.iterrows():
    if 'luad' in str(row['Unnamed: 0']):
        data.at[index, 'is_luad_cancer'] = True
    else:
        data.at[index, 'is_luad_cancer'] = False

# Keep only the rows where 'is_breast_cancer', 'is_prad_cancer', or 'is_luad_cancer' is True
br_data = data[data['is_breast_cancer'] == True]
prad_data = data[data['is_prad_cancer'] == True]
luad_data = data[data['is_luad_cancer'] == True]

# Drop the respective cancer type columns as they are no longer needed
# br_data = br_data.drop(columns=['is_breast_cancer', 'is_prad_cancer', 'is_luad_cancer'])
# prad_data = prad_data.drop(columns=['is_breast_cancer', 'is_prad_cancer', 'is_luad_cancer'])
# luad_data = luad_data.drop(columns=['is_breast_cancer', 'is_prad_cancer', 'is_luad_cancer'])

# Display the datasets after filtering
print("\nDataset after filtering to only display breast cancer:")
print(br_data)

print("\nDataset after filtering to only display prostate cancer:")
print(prad_data)

print("\nDataset after filtering to only display lung cancer:")
print(luad_data)

# Remove rows where all mutation columns are 0
mutation_columns = data.columns[1:]  # Assuming the first column is 'Unnamed: 0'
br_data = br_data[(br_data[mutation_columns] != 0).any(axis=1)]
prad_data = prad_data[(prad_data[mutation_columns] != 0).any(axis=1)]
luad_data = luad_data[(luad_data[mutation_columns] != 0).any(axis=1)]

# Display the datasets after removing rows with all 0s
print("\nBreast cancer dataset after removing rows with all 0s:")
print(br_data)

print("\nProstate cancer dataset after removing rows with all 0s:")
print(prad_data)

print("\nLung cancer dataset after removing rows with all 0s:")
print(luad_data)

# Remove rows where any mutation column has a value higher than 10
br_data = br_data[(br_data[mutation_columns] <= 10).all(axis=1)]
prad_data = prad_data[(prad_data[mutation_columns] <= 10).all(axis=1)]
luad_data = luad_data[(luad_data[mutation_columns] <= 10).all(axis=1)]

# Display the datasets after removing rows with mutation values higher than 10
print("\nBreast cancer dataset after removing rows with mutation values higher than 10:")
print(br_data)

print("\nProstate cancer dataset after removing rows with mutation values higher than 10:")
print(prad_data)

print("\nLung cancer dataset after removing rows with mutation values higher than 10:")
print(luad_data)