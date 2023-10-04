import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from prettytable import PrettyTable
from random import randint
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize


""" Setup """

# Convert images to greyscale numpy arrays
im1 = np.asarray(Image.open(r"beyonce.jpg").convert("L"))
im2 = np.asarray(Image.open(r"einstein.jpg").convert("L"))
im3 = np.asarray(Image.open(r"marie-curie.jpg").convert("L"))
im4 = np.asarray(Image.open(r"michael-jackson.jpg").convert("L"))
im5 = np.asarray(Image.open(r"queen.jpg").convert("L"))

# Clean patterns
clean = np.array([im1, im2, im3, im4, im5])

# Gaussian noise
def awgn(signal, snr):
    db_signal = 10 * np.log10(np.mean(signal ** 2))
    db_noise = db_signal - snr
    noise_power = 10 ** (db_noise / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

# Correct orientation of Ps
noisy = [awgn(index, 20) for index in clean]

# """ Because of the numpy arrays, transposed is actually the correct orientation """


# """ Hebbian Learning Rule """
# make this a function for the next part
# def apply_hebbian_learning_rule(clean_patterns, noisy_patterns):

# Correct orientation of Ts
t_list = [Tx.flatten() for Tx in clean]
t_array = np.array(t_list).T

# Correct orientation of Ps
p_list = [normalize(Px).flatten() for Px in clean]
p_array = np.array(p_list).T

# Correct orientation of noisy Ps
test_p_list = [normalize(tPx).flatten() for tPx in noisy]
test_p_array = np.array(test_p_list).T

# Calculate weight matrix
weight_matrix = np.matmul(t_array, p_array.T)

# Test hebbian learning rule
hebbian_output = [np.matmul(weight_matrix, test_p.T) for test_p in test_p_array.T]


""" Pseudo Inverse Learning Rule """
# make this a function for the next part
# def apply_pseudo_inverse_learning_rule(clean_patterns, noisy_patterns):

# Correct orientation of Ts
t_list = [Tx.flatten() for Tx in clean]
t_array = np.array(t_list).T

# Correct orientation of Ps
p_list = [normalize(Px).flatten() for Px in clean]
p_array = np.array(p_list).T

# Correct orientation of noisy Ps
test_p_list = [normalize(tPx).flatten() for tPx in noisy]
test_p_array = np.array(test_p_list).T

pseudo_inverse_p = np.linalg.pinv(p_array)

# Calculate weight matrix
weight_matrix = np.matmul(t_array, pseudo_inverse_p)

# Test pseudo inverse learning rule
pseudo_inverse_output = [np.matmul(weight_matrix, test_p.T) for test_p in test_p_array.T]


""" Printing """

clean = [Tx.flatten() for Tx in clean]

# Hebbian correlation coefficient table

h_corr11 = pearsonr(clean[0], hebbian_output[0]).statistic
h_corr21 = pearsonr(clean[1], hebbian_output[0]).statistic
h_corr31 = pearsonr(clean[2], hebbian_output[0]).statistic
h_corr41 = pearsonr(clean[3], hebbian_output[0]).statistic
h_corr51 = pearsonr(clean[4], hebbian_output[0]).statistic

h_corr12 = pearsonr(clean[0], hebbian_output[1]).statistic
h_corr22 = pearsonr(clean[1], hebbian_output[1]).statistic
h_corr32 = pearsonr(clean[2], hebbian_output[1]).statistic
h_corr42 = pearsonr(clean[3], hebbian_output[1]).statistic
h_corr52 = pearsonr(clean[4], hebbian_output[1]).statistic

h_corr13 = pearsonr(clean[0], hebbian_output[2]).statistic
h_corr23 = pearsonr(clean[1], hebbian_output[2]).statistic
h_corr33 = pearsonr(clean[2], hebbian_output[2]).statistic
h_corr43 = pearsonr(clean[3], hebbian_output[2]).statistic
h_corr53 = pearsonr(clean[4], hebbian_output[2]).statistic

h_corr14 = pearsonr(clean[0], hebbian_output[3]).statistic
h_corr24 = pearsonr(clean[1], hebbian_output[3]).statistic
h_corr34 = pearsonr(clean[2], hebbian_output[3]).statistic
h_corr44 = pearsonr(clean[3], hebbian_output[3]).statistic
h_corr54 = pearsonr(clean[4], hebbian_output[3]).statistic

h_corr15 = pearsonr(clean[0], hebbian_output[4]).statistic
h_corr25 = pearsonr(clean[1], hebbian_output[4]).statistic
h_corr35 = pearsonr(clean[2], hebbian_output[4]).statistic
h_corr45 = pearsonr(clean[3], hebbian_output[4]).statistic
h_corr55 = pearsonr(clean[4], hebbian_output[4]).statistic

hebbian_table = PrettyTable()
hebbian_table.field_names = ["", "Output 1", "Output 2", "Output 3", "Output 4", "Output 5"]
hebbian_table.add_row(["Pattern 1", h_corr11, h_corr12, h_corr13, h_corr14, h_corr15])
hebbian_table.add_row(["Pattern 2", h_corr21, h_corr22, h_corr23, h_corr24, h_corr25])
hebbian_table.add_row(["Pattern 3", h_corr31, h_corr32, h_corr33, h_corr34, h_corr35])
hebbian_table.add_row(["Pattern 4", h_corr41, h_corr42, h_corr43, h_corr44, h_corr45])
hebbian_table.add_row(["Pattern 5", h_corr51, h_corr52, h_corr53, h_corr54, h_corr55])

print("Hebbian correlation coefficient table")
print(hebbian_table)

# Pseudo inverse correlation coefficient table

pi_corr11 = pearsonr(clean[0], pseudo_inverse_output[0]).statistic
pi_corr21 = pearsonr(clean[1], pseudo_inverse_output[0]).statistic
pi_corr31 = pearsonr(clean[2], pseudo_inverse_output[0]).statistic
pi_corr41 = pearsonr(clean[3], pseudo_inverse_output[0]).statistic
pi_corr51 = pearsonr(clean[4], pseudo_inverse_output[0]).statistic

pi_corr12 = pearsonr(clean[0], pseudo_inverse_output[1]).statistic
pi_corr22 = pearsonr(clean[1], pseudo_inverse_output[1]).statistic
pi_corr32 = pearsonr(clean[2], pseudo_inverse_output[1]).statistic
pi_corr42 = pearsonr(clean[3], pseudo_inverse_output[1]).statistic
pi_corr52 = pearsonr(clean[4], pseudo_inverse_output[1]).statistic

pi_corr13 = pearsonr(clean[0], pseudo_inverse_output[2]).statistic
pi_corr23 = pearsonr(clean[1], pseudo_inverse_output[2]).statistic
pi_corr33 = pearsonr(clean[2], pseudo_inverse_output[2]).statistic
pi_corr43 = pearsonr(clean[3], pseudo_inverse_output[2]).statistic
pi_corr53 = pearsonr(clean[4], pseudo_inverse_output[2]).statistic

pi_corr14 = pearsonr(clean[0], pseudo_inverse_output[3]).statistic
pi_corr24 = pearsonr(clean[1], pseudo_inverse_output[3]).statistic
pi_corr34 = pearsonr(clean[2], pseudo_inverse_output[3]).statistic
pi_corr44 = pearsonr(clean[3], pseudo_inverse_output[3]).statistic
pi_corr54 = pearsonr(clean[4], pseudo_inverse_output[3]).statistic

pi_corr15 = pearsonr(clean[0], pseudo_inverse_output[4]).statistic
pi_corr25 = pearsonr(clean[1], pseudo_inverse_output[4]).statistic
pi_corr35 = pearsonr(clean[2], pseudo_inverse_output[4]).statistic
pi_corr45 = pearsonr(clean[3], pseudo_inverse_output[4]).statistic
pi_corr55 = pearsonr(clean[4], pseudo_inverse_output[4]).statistic

pseudo_inverse_table = PrettyTable()
pseudo_inverse_table.field_names = ["", "Output 1", "Output 2", "Output 3", "Output 4", "Output 5"]
pseudo_inverse_table.add_row(["Pattern 1", pi_corr11, pi_corr12, pi_corr13, pi_corr14, pi_corr15])
pseudo_inverse_table.add_row(["Pattern 2", pi_corr21, pi_corr22, pi_corr23, pi_corr24, pi_corr25])
pseudo_inverse_table.add_row(["Pattern 3", pi_corr31, pi_corr32, pi_corr33, pi_corr34, pi_corr35])
pseudo_inverse_table.add_row(["Pattern 4", pi_corr41, pi_corr42, pi_corr43, pi_corr44, pi_corr45])
pseudo_inverse_table.add_row(["Pattern 5", pi_corr51, pi_corr52, pi_corr53, pi_corr54, pi_corr55])

print("Pseudo inverse correlation coefficient table:")
print(pseudo_inverse_table)


""" Plotting """

rows = 3
cols = 5

# Figure to hold all diagrams
fig = plt.figure(figsize=(cols * 2, rows * 2))

# Add all inputs to figure, reshape them to look correct
fig.add_subplot(rows, cols, 1)
plt.imshow(noisy[0].reshape(64, 64))
fig.add_subplot(rows, cols, 2)
plt.imshow(noisy[1].reshape(64, 64))
fig.add_subplot(rows, cols, 3)
plt.imshow(noisy[2].reshape(64, 64))
fig.add_subplot(rows, cols, 4)
plt.imshow(noisy[3].reshape(64, 64))
fig.add_subplot(rows, cols, 5)
plt.imshow(noisy[4].reshape(64, 64))

# Add all hebbian tests to figure, reshape them to look correct
fig.add_subplot(rows, cols, 6)
plt.imshow(hebbian_output[0].reshape(64, 64))
fig.add_subplot(rows, cols, 7)
plt.imshow(hebbian_output[1].reshape(64, 64))
fig.add_subplot(rows, cols, 8)
plt.imshow(hebbian_output[2].reshape(64, 64))
fig.add_subplot(rows, cols, 9)
plt.imshow(hebbian_output[3].reshape(64, 64))
fig.add_subplot(rows, cols, 10)
plt.imshow(hebbian_output[4].reshape(64, 64))

# Add all pseudo inverse tests to figure, reshape them to look correct
fig.add_subplot(rows, cols, 11)
plt.imshow(pseudo_inverse_output[0].reshape(64, 64))
fig.add_subplot(rows, cols, 12)
plt.imshow(pseudo_inverse_output[1].reshape(64, 64))
fig.add_subplot(rows, cols, 13)
plt.imshow(pseudo_inverse_output[2].reshape(64, 64))
fig.add_subplot(rows, cols, 14)
plt.imshow(pseudo_inverse_output[3].reshape(64, 64))
fig.add_subplot(rows, cols, 15)
plt.imshow(pseudo_inverse_output[4].reshape(64, 64))

plt.show()
