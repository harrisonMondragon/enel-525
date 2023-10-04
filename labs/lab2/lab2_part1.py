import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from random import randint
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize


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

# Calculate weight matrix
weight_matrix = np.matmul(t_array, p_array.T)

# Test hebbian learning rule
hebbian_output = [np.matmul(weight_matrix, test_p.T) for test_p in test_p_array.T]


""" Pseudo Inverse Learning Rule """
# make this a function for the next part
# def apply_pseudo_inverse_learning_rule(clean_patterns, noisy_patterns):

# Correct orientation of Ts
t_array = np.array(clean).T

# Correct orientation of Ps
p_list = [normalize([Px])[0] for Px in clean]
p_array = np.array(p_list).T

# Correct orientation of noisy Ps
test_p_list = [normalize([tPx])[0] for tPx in noisy]
test_p_array = np.array(test_p_list).T

pseudo_inverse_p = np.linalg.pinv(p_array)

# Calculate weight matrix
weight_matrix = np.matmul(t_array, pseudo_inverse_p)

# Test pseudo inverse learning rule
pseudo_inverse_output = [np.matmul(weight_matrix, test_p.T) for test_p in test_p_array.T]


""" Printing """

# Hebbian correlation coefficient table

h_corr11 = pearsonr(clean[0], hebbian_output[0]).statistic
h_corr21 = pearsonr(clean[1], hebbian_output[0]).statistic
h_corr31 = pearsonr(clean[2], hebbian_output[0]).statistic

h_corr12 = pearsonr(clean[0], hebbian_output[1]).statistic
h_corr22 = pearsonr(clean[1], hebbian_output[1]).statistic
h_corr32 = pearsonr(clean[2], hebbian_output[1]).statistic

h_corr13 = pearsonr(clean[0], hebbian_output[2]).statistic
h_corr23 = pearsonr(clean[1], hebbian_output[2]).statistic
h_corr33 = pearsonr(clean[2], hebbian_output[2]).statistic

hebbian_table = PrettyTable()
hebbian_table.field_names = ["", "Output 1", "Output 2", "Output 3"]
hebbian_table.add_row(["Pattern 1", h_corr11, h_corr12, h_corr13])
hebbian_table.add_row(["Pattern 2", h_corr21, h_corr22, h_corr23])
hebbian_table.add_row(["Pattern 3", h_corr31, h_corr32, h_corr33])

print("Hebbian correlation coefficient table")
print(hebbian_table)

# Pseudo inverse correlation coefficient table

pi_corr11 = pearsonr(clean[0], pseudo_inverse_output[0]).statistic
pi_corr21 = pearsonr(clean[1], pseudo_inverse_output[0]).statistic
pi_corr31 = pearsonr(clean[2], pseudo_inverse_output[0]).statistic

pi_corr12 = pearsonr(clean[0], pseudo_inverse_output[1]).statistic
pi_corr22 = pearsonr(clean[1], pseudo_inverse_output[1]).statistic
pi_corr32 = pearsonr(clean[2], pseudo_inverse_output[1]).statistic

pi_corr13 = pearsonr(clean[0], pseudo_inverse_output[2]).statistic
pi_corr23 = pearsonr(clean[1], pseudo_inverse_output[2]).statistic
pi_corr33 = pearsonr(clean[2], pseudo_inverse_output[2]).statistic

pseudo_inverse_table = PrettyTable()
pseudo_inverse_table.field_names = ["", "Output 1", "Output 2", "Output 3"]
pseudo_inverse_table.add_row(["Pattern 1", pi_corr11, pi_corr12, pi_corr13])
pseudo_inverse_table.add_row(["Pattern 2", pi_corr21, pi_corr22, pi_corr23])
pseudo_inverse_table.add_row(["Pattern 3", pi_corr31, pi_corr32, pi_corr33])

print("Pseudo inverse correlation coefficient table:")
print(pseudo_inverse_table)

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

# Add all hebbian tests to figure, reshape them to look correct
fig.add_subplot(rows, cols, 4)
plt.imshow(hebbian_output[0].reshape(6, 5))
fig.add_subplot(rows, cols, 5)
plt.imshow(hebbian_output[1].reshape(6, 5))
fig.add_subplot(rows, cols, 6)
plt.imshow(hebbian_output[2].reshape(6, 5))

# Add all pseudo inverse tests to figure, reshape them to look correct
fig.add_subplot(rows, cols, 7)
plt.imshow(pseudo_inverse_output[0].reshape(6, 5))
fig.add_subplot(rows, cols, 8)
plt.imshow(pseudo_inverse_output[1].reshape(6, 5))
fig.add_subplot(rows, cols, 9)
plt.imshow(pseudo_inverse_output[2].reshape(6, 5))

plt.show()
