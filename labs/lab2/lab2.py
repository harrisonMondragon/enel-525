import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from prettytable import PrettyTable


""" Setup """

# 0 digit pattern
P1 = np.array([ 1, -1, -1, -1,  1,
               -1,  1,  1,  1, -1,
               -1,  1,  1,  1, -1,
               -1,  1,  1,  1, -1,
               -1,  1,  1,  1, -1,
                1, -1, -1, -1,  1])

# 1 digit pattern
P2 = np.array([ 1, -1, -1,  1,  1,
                1,  1, -1,  1,  1,
                1,  1, -1,  1,  1,
                1,  1, -1,  1,  1,
                1,  1, -1,  1,  1,
                1,  1, -1,  1,  1,])

# 2 digit pattern
P3 = np.array([-1, -1, -1,  1,  1,
                1,  1,  1, -1,  1,
                1,  1,  1, -1,  1,
                1, -1, -1,  1,  1,
                1, -1,  1,  1,  1,
                1, -1, -1, -1, -1])

# Clean patterns
clean = np.array([P1, P2, P3])

# Noisy patterns
noisy = np.copy(clean)
for index in noisy:
    for i in range(3):
        flip_pixel = randint(0, 29)
        index[flip_pixel] *= -1

""" Because of the numpy arrays, transposed is actually the correct orientation """

""" Hebbian Learning Rule """
# make this a function for the next part
# def apply_hebbian_learning_rule(clean_patterns, noisy_patterns):

# Correct orientation of Ts
t_array = np.array(clean).T

# Correct orientation of Ps
p_list = [normalize([Px])[0] for Px in clean]
p_array = np.array(p_list).T

# Correct orientation of noisy Ps
test_p_list = [normalize([tPx])[0] for tPx in noisy]
test_p_array = np.array(test_p_list).T

# print(f"t_array shape {t_array.shape}")
# print(f"p_array shape {p_array.shape}")
# print(f"test_p_array shape {test_p_array.shape}")

# Calculate weight matrix
weight_matrix = np.matmul(t_array, p_array.T)

# Test hebbian learning rule ------------ Something is wrong here!!!!!!
hebbian_output = [np.matmul(weight_matrix, test_p) for test_p in test_p_array.T]


# """ Pseudo Inverse Learning Rule """
# # make this a function for the next part
# # def apply_pseudo_inverse_learning_rule(clean_patterns, noisy_patterns):

# # Correct orientation of noisy Ps
# test_p_array = [normalize(tPx).T for tPx in noisy]

# # Input matrix
# input_matrix = np.array([clean])

# # Calculate pseudo inverse matrix
# pseudo_inverse = np.linalg.pinv(input_matrix)
# print(f"Shape of PI: {pseudo_inverse.shape}")

# # Calculate weight matrix
# weight_matrix = np.matmul(input_matrix, pseudo_inverse)
# print(f"Shape of weight: {weight_matrix.shape}")


""" Plotting """

rows = 3
cols = 3

# Figure to hold all diagrams
fig = plt.figure(figsize=(7, 7))

# Add all inputs to figure, reshape them to look correct
fig.add_subplot(rows, cols, 1)
plt.imshow(clean[0].reshape(6, 5))
fig.add_subplot(rows, cols, 2)
plt.imshow(clean[1].reshape(6, 5))
fig.add_subplot(rows, cols, 3)
plt.imshow(clean[2].reshape(6, 5))

# Add all noisy inputs to figure, reshape them to look correct
fig.add_subplot(rows, cols, 4)
plt.imshow(noisy[0].reshape(6, 5))
fig.add_subplot(rows, cols, 5)
plt.imshow(noisy[1].reshape(6, 5))
fig.add_subplot(rows, cols, 6)
plt.imshow(noisy[2].reshape(6, 5))

# Add all hebbian tests to figure, reshape them to look correct
fig.add_subplot(rows, cols, 7)
plt.imshow(hebbian_output[0].reshape(6, 5))
fig.add_subplot(rows, cols, 8)
plt.imshow(hebbian_output[1].reshape(6, 5))
fig.add_subplot(rows, cols, 9)
plt.imshow(hebbian_output[2].reshape(6, 5))

plt.show()

# # """ Printing """

# # # P1_Hebbian = pearsonr(clean[0][0], hebbian_output[0][0])
# # # P2_Hebbian = pearsonr(clean[1][0], hebbian_output[1][0])
# # # P3_Hebbian = pearsonr(clean[2][0], hebbian_output[2][0])

# # # correlation_coefficient_table = PrettyTable()
# # # correlation_coefficient_table.field_names = ["Pattern", "Hebbian", "Pseudo Inverse"]
# # # correlation_coefficient_table.add_row(["Gridwise 0 (P1)", P1_Hebbian.statistic, -1])
# # # correlation_coefficient_table.add_row(["Gridwise 1 (P2)", P2_Hebbian.statistic, -1])
# # # correlation_coefficient_table.add_row(["Gridwise 2 (P3)", P3_Hebbian.statistic, -1])

# # # print(correlation_coefficient_table)
