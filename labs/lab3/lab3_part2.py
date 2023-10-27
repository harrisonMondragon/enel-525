from PIL import Image
import numpy as np

img1 = np.asarray(Image.open('Lab3-chars/char_a.bmp')).reshape(400,)
img2 = np.asarray(Image.open('Lab3-chars/char_b.bmp')).reshape(400,)
img3 = np.asarray(Image.open('Lab3-chars/char_c.bmp')).reshape(400,)
img4 = np.asarray(Image.open('Lab3-chars/char_d.bmp')).reshape(400,)
img5 = np.asarray(Image.open('Lab3-chars/char_e.bmp')).reshape(400,)
img6 = np.asarray(Image.open('Lab3-chars/char_f.bmp')).reshape(400,)
img7 = np.asarray(Image.open('Lab3-chars/char_g.bmp')).reshape(400,)
img8 = np.asarray(Image.open('Lab3-chars/char_h.bmp')).reshape(400,)
img9 = np.asarray(Image.open('Lab3-chars/char_i.bmp')).reshape(400,)
img10 = np.asarray(Image.open('Lab3-chars/char_j.bmp')).reshape(400,)
img11 = np.asarray(Image.open('Lab3-chars/char_k.bmp')).reshape(400,)
img12 = np.asarray(Image.open('Lab3-chars/char_l.bmp')).reshape(400,)
img13 = np.asarray(Image.open('Lab3-chars/char_m.bmp')).reshape(400,)
img14 = np.asarray(Image.open('Lab3-chars/char_n.bmp')).reshape(400,)
img15 = np.asarray(Image.open('Lab3-chars/char_o.bmp')).reshape(400,)
img16 = np.asarray(Image.open('Lab3-chars/char_p.bmp')).reshape(400,)
img17 = np.asarray(Image.open('Lab3-chars/char_q.bmp')).reshape(400,)
img18 = np.asarray(Image.open('Lab3-chars/char_r.bmp')).reshape(400,)
img19 = np.asarray(Image.open('Lab3-chars/char_s.bmp')).reshape(400,)
img20 = np.asarray(Image.open('Lab3-chars/char_t.bmp')).reshape(400,)
img21 = np.asarray(Image.open('Lab3-chars/char_u.bmp')).reshape(400,)
img22 = np.asarray(Image.open('Lab3-chars/char_v.bmp')).reshape(400,)
img23 = np.asarray(Image.open('Lab3-chars/char_w.bmp')).reshape(400,)
img24 = np.asarray(Image.open('Lab3-chars/char_x.bmp')).reshape(400,)
img25 = np.asarray(Image.open('Lab3-chars/char_y.bmp')).reshape(400,)
img26 = np.asarray(Image.open('Lab3-chars/char_z.bmp')).reshape(400,)







import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

# Given mean squared error function
def mse(array):
    return np.mean(array.flatten() ** 2)

# T and P values, T values were arbitraily assigned per class
p_array = np.array([[1,1],[1,2],[2,-1],[2,0],[-1,2],[-2,1],[-1,-1],[-2,-2]]).T
t_array = np.array([[-1,-1],[-1,-1],[-1,1],[-1,1],[1,-1],[1,-1],[1,1],[1,1]]).T

# Initial weights and baiases
W = np.zeros((2,2))
b = np.zeros((2,1))

# Learing rate
alpha = 0.01

# Error variable to hold the current iteration's mse
mse_value = 999

# List that holds all error values intill the error threshold has been acheived
mse_list = []

# for number in range(100):
while mse_value > 0.02:
    errors = []
    for i in range(len(p_array)):
        a = np.dot(W, p_array[:, [i]]) + b
        error = t_array[:, [i]] - a

        W = W + 2 * alpha * np.dot(error, p_array[:, [i]].T)
        b = b + 2 * alpha * error

        errors.append(error)

    mse_value = mse(np.array(errors))
    mse_list.append(mse_value)

# print(mse_list)

plt.semilogy(mse_list)
plt.xlabel("Iteration")
plt.ylabel("MSE Value")
plt.show()