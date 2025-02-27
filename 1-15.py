# Name: Thomas Miller
# This script:
# - Uses the Iris dataset to create a scatter plot of the data
# - Saves the plot as a PNG file
# - Calculates and plots the average values for each class
# - Calculates and plots the line of best fit for each class
# - Prints the count of each class to the console
# - Prints the equation of the line of best fit for each class to the console
# - Prints the average values for each class to the console
from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
list(iris.target_names)
print(iris.target_names)
setosa_count, versicolor_count, virginica_count = 0, 0, 0

for target in iris.target:
    if target == 0:
        setosa_count += 1
    elif target == 1:
        versicolor_count += 1
    elif target == 2:
        virginica_count += 1

print("Setosa count: ", setosa_count)
print("Versicolor count: ", versicolor_count)
print("Virginica count: ", virginica_count)

import matplotlib.pyplot as plt

setosa_data = np.array([iris.data[i] for i in range(len(iris.target)) if iris.target[i] == 0])
versicolor_data = np.array([iris.data[i] for i in range(len(iris.target)) if iris.target[i] == 1])
virginica_data = np.array([iris.data[i] for i in range(len(iris.target)) if iris.target[i] == 2])

plt.figure()

# Plot Setosa data
plt.scatter(setosa_data[:, 0], setosa_data[:, 1], label='Setosa', color='purple')
mean_setosa_x = np.mean(setosa_data[:, 0])
mean_setosa_y = np.mean(setosa_data[:, 1])
plt.plot(mean_setosa_x, mean_setosa_y, 'mo')
print(f"Setosa average: ({mean_setosa_x}, {mean_setosa_y})")

# Line of best fit for Setosa
m, b = np.polyfit(setosa_data[:, 0], setosa_data[:, 1], 1)
plt.plot(setosa_data[:, 0], m*setosa_data[:, 0] + b, 'm-')
print(f"Setosa line of best fit: y = {m}x + {b}")


# Plot Versicolor data
plt.scatter(versicolor_data[:, 0], versicolor_data[:, 1], label='Versicolor', color='orange')
mean_versicolor_x = np.mean(versicolor_data[:, 0])
mean_versicolor_y = np.mean(versicolor_data[:, 1])
plt.plot(mean_versicolor_x, mean_versicolor_y, 'yo')
print(f"Versicolor average: ({mean_versicolor_x}, {mean_versicolor_y})")

# Line of best fit for Versicolor
m, b = np.polyfit(versicolor_data[:, 0], versicolor_data[:, 1], 1)
plt.plot(versicolor_data[:, 0], m*versicolor_data[:, 0] + b, 'y-')
print(f"Versicolor line of best fit: y = {m}x + {b}")



# Plot Virginica data
plt.scatter(virginica_data[:, 0], virginica_data[:, 1], label='Virginica', color='blue')
mean_virginica_x = np.mean(virginica_data[:, 0])
mean_virginica_y = np.mean(virginica_data[:, 1])
plt.plot(mean_virginica_x, mean_virginica_y, 'co')
print(f"Virginica average: ({mean_virginica_x}, {mean_virginica_y})")

# Line of best fit for Virginica
m, b = np.polyfit(virginica_data[:, 0], virginica_data[:, 1], 1)
plt.plot(virginica_data[:, 0], m*virginica_data[:, 0] + b, 'c-')
print(f"Virginica line of best fit: y = {m}x + {b}")


plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title('Iris Data with Average Points')

plt.savefig('1-15.png')