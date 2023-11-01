import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from prettytable import PrettyTable
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize

# Given mean squared error function
def mse(array):
    return np.mean(array.flatten() ** 2)

# Convert images to numpy arrays
img1 = np.asarray(Image.open(r"lab3_chars/char_a.bmp"))
img2 = np.asarray(Image.open(r"lab3_chars/char_b.bmp"))
img3 = np.asarray(Image.open(r"lab3_chars/char_c.bmp"))
img4 = np.asarray(Image.open(r"lab3_chars/char_d.bmp"))
img5 = np.asarray(Image.open(r"lab3_chars/char_e.bmp"))
img6 = np.asarray(Image.open(r"lab3_chars/char_f.bmp"))
img7 = np.asarray(Image.open(r"lab3_chars/char_f.bmp"))
img8 = np.asarray(Image.open(r"lab3_chars/char_h.bmp"))
img9 = np.asarray(Image.open(r"lab3_chars/char_i.bmp"))
img10 = np.asarray(Image.open(r"lab3_chars/char_j.bmp"))
img11 = np.asarray(Image.open(r"lab3_chars/char_k.bmp"))
img12 = np.asarray(Image.open(r"lab3_chars/char_l.bmp"))
img13 = np.asarray(Image.open(r"lab3_chars/char_m.bmp"))
img14 = np.asarray(Image.open(r"lab3_chars/char_n.bmp"))
img15 = np.asarray(Image.open(r"lab3_chars/char_o.bmp"))
img16 = np.asarray(Image.open(r"lab3_chars/char_p.bmp"))
img17 = np.asarray(Image.open(r"lab3_chars/char_q.bmp"))
img18 = np.asarray(Image.open(r"lab3_chars/char_r.bmp"))
img19 = np.asarray(Image.open(r"lab3_chars/char_s.bmp"))
img20 = np.asarray(Image.open(r"lab3_chars/char_t.bmp"))
img21 = np.asarray(Image.open(r"lab3_chars/char_u.bmp"))
img22 = np.asarray(Image.open(r"lab3_chars/char_v.bmp"))
img23 = np.asarray(Image.open(r"lab3_chars/char_w.bmp"))
img24 = np.asarray(Image.open(r"lab3_chars/char_x.bmp"))
img25 = np.asarray(Image.open(r"lab3_chars/char_y.bmp"))
img26 = np.asarray(Image.open(r"lab3_chars/char_z.bmp"))

input_list = [  img1, img2, img3, img4, img5, img6, img7,
                img8, img9, img10, img11, img12, img13, img14,
                img15, img16, img17, img18, img19, img20, img21,
                img22, img23, img24, img23, img24,]


# Since it is an autoassociaton network, Px = normalize(Tx)
t_list = [Tx.reshape(400,1) for Tx in input_list]
p_list = [normalize(Px) for Px in t_list]

t_list = np.array(t_list)
p_list = np.array(p_list)

# print(f"t_list[0].shape {t_list[0].shape}")

# Initial weight and baias
W = np.zeros((400,400))
b = np.zeros((400,1))

# Learing rate
alpha = 0.002

# Error variable to hold the current iteration's mse
mse_value = 999

# List that holds all error values until the error threshold has been acheived
mse_list = []

# "MSE" loop
for number in range(300):
# while mse_value > 0.00001:
    errors = []

    # "Iteration" loop
    for i in range(len(p_list)):

        # Calculate output and error
        a = W.dot(p_list[i]) + b
        e = p_list[i] - a

        # Update W and b
        W = W + 2 * alpha * e.dot(p_list[i].T)
        b = b + 2 * alpha * e

        errors.append(e)

        # print(f"a.shape {a.shape}")
        # print(f"e.shape {e.shape}")

    # Calculate mse and add it to the mse list
    mse_value = mse(np.array(errors))
    mse_list.append(mse_value)

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
trained_list = np.array(trained_list)

t_list = np.squeeze(t_list)
trained_list = np.squeeze(trained_list)

# Calculate correlations
corr_table = PrettyTable()
for i in range(len(t_list)):
    column_list = []
    for j in range(len(trained_list)):
        column_list.append(pearsonr(t_list[i, :], trained_list[j, :]).statistic)

    corr_table.add_column(f"Input{i}", column_list)

print(corr_table)