import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Creating the dataset
num_samples = 1000
num_ft = 2

"""--------Exercise 3--------"""
adult = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
child = [10,5]*np.random.randn(num_samples,num_ft) + [50,25]
dataset_array = np.concatenate((adult, child), axis=0)

target_1 = np.zeros(1000)
target_2 = np.ones(1000)
targets_array = np.concatenate((target_1, target_2), axis=0)

# Scatter plot
classes = ["Adults", "Children"]

scatter_plot = plt.scatter(dataset_array[:,0],dataset_array[:,1],c=targets_array)

plt.legend(handles=scatter_plot.legend_elements()[0], labels=classes)
plt.title("Lesson 1 Exercise 3 Dataset")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()

# Convert to dataframes
dataset = pd.DataFrame(dataset_array)
targets = pd.DataFrame(targets_array)

# Export as csv files
dataset.to_csv("dataset.csv", index=False)
targets.to_csv("targets.csv", index=False)

# Read csv files
check_dataset = pd.read_csv("dataset.csv")
check_targets = pd.read_csv("targets.csv")

# Print the read in csv files
print(f"Dataset read from csv file:\n\n{check_dataset}")
print(f"Targets read from csv file:\n\n{check_targets}")
