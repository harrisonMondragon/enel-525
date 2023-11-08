import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# Given mean squared error function
def mse(array):
    return np.mean(array.flatten() ** 2)

# Creating the data sequence
data_sequence = np.load(r"data1.npy")

# Initial weight and baias
W1 = np.random.normal(size=(5, 2))
b1 = np.random.normal(size=(5, 1))

W2 = np.random.normal(size=(1, 5))
b2 = np.random.normal(size=(1, 1))

# Learing rate and error threshold
alpha = 0.05
error_threshold = 0.00002

# Error variable to hold the current iteration's mse
mse_value = 0.5

# List that holds all error values until the error threshold has been acheived
mse_list = []

# "MSE" loop
while mse_value > error_threshold:
    errors = []

    i = 1

    while i < 169:

        # Get p from previous 2 points, p = a0
        a0 = np.array(np.array([data_sequence[i],data_sequence[i-1]]))

        # Calculate a1
        a1 = W1.dot(a0) + b1
        for j in range(5):
            a1[j][0] = 1/(1 + np.exp(-a1[j][0]))

        # Calculate a2
        a2 = W2.dot(a1) + b2

        # Calculate e and append to the iteration error list
        e = data_sequence[i+1] - a2
        errors.append(e)

        # Calculate s2
        s2 = -2 * e

        # Calculate s1
        F1n1 = np.zeros((5,5))
        for k in range(5):
            F1n1[k][k] = (1-a1[k][0]) * a1[k][0]
        s1 = F1n1.dot(W2.T) * s2

        # Update W2 and b2
        W2 = W2 - alpha * s2.dot(a1.T)
        b2 = b2 - alpha * s2

        # Update W1 and b1
        W1 = W1 - alpha * s1.dot(a0.T)
        b1 = b1 - alpha * s1

        i += 1

    # # Calculate mse and add it to the mse list
    mse_value = mse(np.array(errors))
    mse_list.append(mse_value)

# print(f"Final W1 {W1}")
# print(f"Final b1 {b1}")
# print(f"Final W2 {W2}")
# print(f"Final b2 {b2}")

# Plot the learning curve
plt.semilogy(mse_list)
plt.xlabel("Iteration")
plt.ylabel("MSE Value")
plt.show()


# Compare predicted and true values
predicted_values = []
g = 169
while g < 179:
    # Get p from previous 2 points, p = a0
    a0 = np.array(np.array([data_sequence[g],data_sequence[g-1]]))

    # Calculate a1
    a1 = W1.dot(a0) + b1
    for j in range(5):
        a1[j][0] = 1/(1 + np.exp(-a1[j][0]))

    # Calculate a2
    a2 = W2.dot(a1) + b2
    predicted_values.append(a2[0][0])

    g += 1

true_values = data_sequence[170:]

# Table format
display_table = PrettyTable()
display_table.add_column("Predicted", predicted_values)
display_table.add_column("True", true_values)
print(display_table)

# Plot format
plt.figure("ahh")
plt.plot(predicted_values)
plt.plot(true_values)
plt.show()