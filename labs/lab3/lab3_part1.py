import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

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

trained_list = []
for j in range(len(p_array)):
    trained_list.append(np.matmul(W, p_array[j]) + b)

mse11 = mse(t_array[0]-trained_list[0])
mse12 = mse(t_array[0]-trained_list[1])
mse13 = mse(t_array[0]-trained_list[2])
mse14 = mse(t_array[0]-trained_list[3])
mse15 = mse(t_array[0]-trained_list[4])
mse16 = mse(t_array[0]-trained_list[5])
mse17 = mse(t_array[0]-trained_list[6])
mse18 = mse(t_array[0]-trained_list[7])

mse21 = mse(t_array[2]-trained_list[0])
mse22 = mse(t_array[2]-trained_list[1])
mse23 = mse(t_array[2]-trained_list[2])
mse24 = mse(t_array[2]-trained_list[3])
mse25 = mse(t_array[2]-trained_list[4])
mse26 = mse(t_array[2]-trained_list[5])
mse27 = mse(t_array[2]-trained_list[6])
mse28 = mse(t_array[2]-trained_list[7])

mse31 = mse(t_array[4]-trained_list[0])
mse32 = mse(t_array[4]-trained_list[1])
mse33 = mse(t_array[4]-trained_list[2])
mse34 = mse(t_array[4]-trained_list[3])
mse35 = mse(t_array[4]-trained_list[4])
mse36 = mse(t_array[4]-trained_list[5])
mse37 = mse(t_array[4]-trained_list[6])
mse38 = mse(t_array[4]-trained_list[7])

mse41 = mse(t_array[6]-trained_list[0])
mse42 = mse(t_array[6]-trained_list[1])
mse43 = mse(t_array[6]-trained_list[2])
mse44 = mse(t_array[6]-trained_list[3])
mse45 = mse(t_array[6]-trained_list[4])
mse46 = mse(t_array[6]-trained_list[5])
mse47 = mse(t_array[6]-trained_list[6])
mse48 = mse(t_array[6]-trained_list[7])

mse_table = PrettyTable()
mse_table.field_names = ["", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
mse_table.add_row(["Class1", mse11,mse12,mse13,mse14,mse15,mse16,mse17,mse18])
mse_table.add_row(["Class2", mse21,mse22,mse23,mse24,mse25,mse26,mse27,mse28])
mse_table.add_row(["Class3", mse31,mse32,mse33,mse34,mse35,mse36,mse37,mse38])
mse_table.add_row(["Class4", mse41,mse42,mse43,mse44,mse45,mse46,mse47,mse48])

print(mse_table)