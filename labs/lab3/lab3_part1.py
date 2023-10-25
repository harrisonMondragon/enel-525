import matplotlib.pyplot as plt
import numpy as np

# Given mean squared error function
def mse(array):
    return np.mean(array.flatten() ** 2)

# T and P values, T values were arbitraily assigned per class
p_array = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]])
t_array = np.array([[-1,-1],[-1,-1],[-1,1],[-1,1],[1,-1],[1,-1],[1,1],[1,1]])

# Initial weights and baiases
W = np.array([[1,0],[0,1]])
b = np.array([1,1])

# Learing rate
alpha = 0.04

# Error array to fill per iteration to create the input for the mse function
mse_input = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

# Error variable to hold the current iteration's mse
mse_value = 999

# List that holds all error values intill the error threshold has been acheived
mse_list = []

while mse_value >= 4.4376:
    for i in range(len(p_array)):
        error = t_array[i] - (np.matmul(W, p_array[i]) + b)

        W = W + 2 * alpha * np.matmul(error, p_array[i].T)
        b = b + 2 * alpha * error

        mse_input[i] = error

    mse_value = mse(mse_input)
    mse_list.append(mse_value)

print(mse_list)

plt.semilogy(mse_list)
plt.xlabel("Iteration")
plt.ylabel("MSE Value")
plt.show()