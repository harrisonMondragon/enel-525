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

# Noisy patterns
noisy = np.copy(clean)
for index in noisy:
    db_signal = 10 * np.log10(np.mean(index ** 2))
    db_noise = db_signal - 20
    noise_power = 10 ** (db_noise / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(index))
    index = index + noise



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


""" Plotting """

rows = 3
cols = 5

# Figure to hold all diagrams
fig = plt.figure(figsize=(10, 6))

# Add all inputs to figure, reshape them to look correct
fig.add_subplot(rows, cols, 1)
plt.imshow(clean[0].reshape(64, 64))
fig.add_subplot(rows, cols, 2)
plt.imshow(clean[1].reshape(64, 64))
fig.add_subplot(rows, cols, 3)
plt.imshow(clean[2].reshape(64, 64))
fig.add_subplot(rows, cols, 4)
plt.imshow(clean[3].reshape(64, 64))
fig.add_subplot(rows, cols, 5)
plt.imshow(clean[4].reshape(64, 64))

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