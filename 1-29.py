import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'dataset.csv'
data = pd.read_csv(file_path)

mutation_columns = data.columns[1:]

# Remove rows with more than 6 mutations
data = data[(data[mutation_columns] <= 6).all(axis=1)]

# Filter to only have breast cancer, prad cancer, or luad cancer data
br_data = data[data['Unnamed: 0'].str.contains('brca', na=False)]
prad_data = data[data['Unnamed: 0'].str.contains('prad', na=False)]
luad_data = data[data['Unnamed: 0'].str.contains('luad', na=False)]

# Function to reserve a percentage of data
def reserve_percentage(data, percentage):
    return data.sample(frac=percentage, random_state=1)

# Reserve X% of the data for each type of cancer
reserved_percentage = 0.1
reserved_br_data = reserve_percentage(br_data, reserved_percentage)
reserved_prad_data = reserve_percentage(prad_data, reserved_percentage)
reserved_luad_data = reserve_percentage(luad_data, reserved_percentage)

# Combine reserved data into a single dataframe
reserved_data = pd.concat([reserved_br_data, reserved_prad_data, reserved_luad_data])

# Combine leftover data into a single dataframe
leftover_br_data = br_data.drop(reserved_br_data.index)
leftover_prad_data = prad_data.drop(reserved_prad_data.index)
leftover_luad_data = luad_data.drop(reserved_luad_data.index)

training_data = pd.concat([leftover_br_data, leftover_prad_data, leftover_luad_data])

training_data_copy1 = training_data.copy()
training_data_copy2 = training_data.copy()
training_data_copy3 = training_data.copy()

training_data_copy1['label'] = training_data_copy1['Unnamed: 0'].apply(lambda x: 1 if 'brca' in x else 0)
training_data_copy2['label'] = training_data_copy2['Unnamed: 0'].apply(lambda x: 1 if 'prad' in x else 0)
training_data_copy3['label'] = training_data_copy3['Unnamed: 0'].apply(lambda x: 1 if 'luad' in x else 0)

def run_logistic_regression(data):
    X = data[mutation_columns]
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

model1, accuracy1 = run_logistic_regression(training_data_copy1)
model2, accuracy2 = run_logistic_regression(training_data_copy2)
model3, accuracy3 = run_logistic_regression(training_data_copy3)

print(f'Accuracy for BRCA model: {accuracy1}')
print(f'Accuracy for PRAD model: {accuracy2}')
print(f'Accuracy for LUAD model: {accuracy3}')

reserved_data_copy1 = reserved_data.copy()
reserved_data_copy2 = reserved_data.copy()
reserved_data_copy3 = reserved_data.copy()

reserved_data_copy1['label'] = reserved_data_copy1['Unnamed: 0'].apply(lambda x: 1 if 'brca' in x else 0)
reserved_data_copy2['label'] = reserved_data_copy2['Unnamed: 0'].apply(lambda x: 1 if 'prad' in x else 0)
reserved_data_copy3['label'] = reserved_data_copy3['Unnamed: 0'].apply(lambda x: 1 if 'luad' in x else 0)

def test_model(model, data):
    X = data[mutation_columns]
    y = data['label']
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy

reserved_accuracy1 = test_model(model1, reserved_data_copy1)
reserved_accuracy2 = test_model(model2, reserved_data_copy2)
reserved_accuracy3 = test_model(model3, reserved_data_copy3)

print(f'Reserved accuracy for BRCA model: {reserved_accuracy1}')
print(f'Reserved accuracy for PRAD model: {reserved_accuracy2}')
print(f'Reserved accuracy for LUAD model: {reserved_accuracy3}')