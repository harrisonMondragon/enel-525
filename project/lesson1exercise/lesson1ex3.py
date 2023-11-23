"""--------Exercise 2--------"""

import numpy as np
import matplotlib.pyplot as plt

# # Creating the dataset
num_samples = 1000
num_ft = 2

# a = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
# print(a.shape)

# # Calculating the mean and std
# m = np.mean(a, axis=0)
# st = np.std(a, axis=0)
# print(f"Mean = {m}")
# print(f"std = {st}")

# # Scatter plot
# plt.scatter(a[:,0],a[:,1])
# plt.title("Dataset a")
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.show()

# # Histogram plot
# plt.hist(a[:,0])
# plt.title("Height Histogram")
# plt.xlabel("Height")
# plt.ylabel("Number of Samples")
# plt.show()

# # Standardizing the dataset
# standard_a = (a-m)/st

# plt.scatter(standard_a[:,0],standard_a[:,1])
# plt.title("Standardized Dataset a")
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.show()


"""--------Exercise 3--------"""
#    std                      dimensions        normal dist
# a = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
adult = [10,5]*np.random.randn(num_samples,num_ft) + [150,50]
child = [10,5]*np.random.randn(num_samples,num_ft) + [50,25]
dataset_array = np.concatenate((adult, child), axis=0)

target_1 = np.zeros(1000)
target_2 = np.ones(1000)
targets_array = np.concatenate((target_1, target_2), axis=0)

# Scatter plot
plt.scatter(dataset_array[:,0],dataset_array[:,1],c=targets_array)
plt.title("Lesson 1 Exercise 3 Dataset")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()
