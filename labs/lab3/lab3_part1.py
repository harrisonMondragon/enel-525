import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# Given mean squared error function
def mse(array):
    return np.mean(array.flatten() ** 2)

# T and P values, T values were arbitraily assigned per class
p_list =    [   np.array([[1,1]]).T, np.array([[1,2]]).T,
                np.array([[2,-1]]).T, np.array([[2,0]]).T,
                np.array([[-1,2]]).T, np.array([[-2,1]]).T,
                np.array([[-1,-1]]).T, np.array([[-2,-2]]).T
            ]

t_list =    [   np.array([[-1,-1]]).T, np.array([[-1,-1]]).T,
                np.array([[-1,1]]).T, np.array([[-1,1]]).T,
                np.array([[1,-1]]).T, np.array([[1,-1]]).T,
                np.array([[1,1]]).T, np.array([[1,1]]).T
            ]

# Initial weight and baias
W = np.zeros((2,2))
b = np.zeros((2,1))

# Learing rate
alpha = 0.01

# Error variable to hold the current iteration's mse
mse_value = 999

# List that holds all error values until the error threshold has been acheived
mse_list = []

# "MSE" loop
while mse_value > 0.1208:
    errors = []

    # "Iteration" loop
    for i in range(len(p_list)):

        # Calculate output and error
        a = W.dot(p_list[i]) + b
        e = t_list[i] - a

        # Update W and b
        W = W + 2 * alpha * e.dot(p_list[i].T)
        b = b + 2 * alpha * e

        errors.append(e)

    # Calculate mse and add it to the mse list
    mse_value = mse(np.array(errors))
    mse_list.append(mse_value)

print(f"Final W {W}")
print(f"Final b {b}")

# Plot the learning curve
plt.semilogy(mse_list)
plt.xlabel("Iteration")
plt.ylabel("MSE Value")
plt.show()


# Calculate outputs using trained weight and bias
trained_list = []
for i in range(len(p_list)):
    a = W.dot(p_list[i]) + b
    trained_list.append(a)

# Calculate table entries
mse11 = round(mse(t_list[0]-trained_list[0]),6)
mse12 = round(mse(t_list[0]-trained_list[1]),6)
mse13 = round(mse(t_list[0]-trained_list[2]),6)
mse14 = round(mse(t_list[0]-trained_list[3]),6)
mse15 = round(mse(t_list[0]-trained_list[4]),6)
mse16 = round(mse(t_list[0]-trained_list[5]),6)
mse17 = round(mse(t_list[0]-trained_list[6]),6)
mse18 = round(mse(t_list[0]-trained_list[7]),6)

mse21 = round(mse(t_list[2]-trained_list[0]),6)
mse22 = round(mse(t_list[2]-trained_list[1]),6)
mse23 = round(mse(t_list[2]-trained_list[2]),6)
mse24 = round(mse(t_list[2]-trained_list[3]),6)
mse25 = round(mse(t_list[2]-trained_list[4]),6)
mse26 = round(mse(t_list[2]-trained_list[5]),6)
mse27 = round(mse(t_list[2]-trained_list[6]),6)
mse28 = round(mse(t_list[2]-trained_list[7]),6)

mse31 = round(mse(t_list[4]-trained_list[0]),6)
mse32 = round(mse(t_list[4]-trained_list[1]),6)
mse33 = round(mse(t_list[4]-trained_list[2]),6)
mse34 = round(mse(t_list[4]-trained_list[3]),6)
mse35 = round(mse(t_list[4]-trained_list[4]),6)
mse36 = round(mse(t_list[4]-trained_list[5]),6)
mse37 = round(mse(t_list[4]-trained_list[6]),6)
mse38 = round(mse(t_list[4]-trained_list[7]),6)

mse41 = round(mse(t_list[6]-trained_list[0]),6)
mse42 = round(mse(t_list[6]-trained_list[1]),6)
mse43 = round(mse(t_list[6]-trained_list[2]),6)
mse44 = round(mse(t_list[6]-trained_list[3]),6)
mse45 = round(mse(t_list[6]-trained_list[4]),6)
mse46 = round(mse(t_list[6]-trained_list[5]),6)
mse47 = round(mse(t_list[6]-trained_list[6]),6)
mse48 = round(mse(t_list[6]-trained_list[7]),6)

# Format table
mse_table = PrettyTable()
mse_table.field_names = ["", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
mse_table.add_row(["Class1", mse11,mse12,mse13,mse14,mse15,mse16,mse17,mse18])
mse_table.add_row(["Class2", mse21,mse22,mse23,mse24,mse25,mse26,mse27,mse28])
mse_table.add_row(["Class3", mse31,mse32,mse33,mse34,mse35,mse36,mse37,mse38])
mse_table.add_row(["Class4", mse41,mse42,mse43,mse44,mse45,mse46,mse47,mse48])

print(mse_table)