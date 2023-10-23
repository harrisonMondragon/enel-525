# Didn't save my lab1 by accident, this is John's incase I need to look at it later

# ENEL 525 Lab 1
# Author: John McMurtry

import numpy
import matplotlib.pyplot as plt

#p = numpy.array([[1, -1, 0],[2, 2, -1]])
#t = numpy.array([1, 0, 0])

p = numpy.array([[1, 2, 3, 1, 2, 4], 
                [4, 5, 3.5, 0.5, 2, 0.5]])
t = numpy.array([1, 1, 1, 0, 0, 0])

def perceptron(p , t):

    a = 0
    w = numpy.array([0, 0])
    b = 0
    e = 0
    output = [0]*len(p[0])
    flag = [1]*len(p[0])
    class1 = ([], [])
    class2 = ([], [])

    while(any(flag)):

        flag = [1]*len(p[0])
        i = 0

        while i < len(flag):

            vector = numpy.array([p[0,i], p[1,i]])
            inner = (numpy.dot(w, vector) + b)

            if(inner >= 0):
                a = 1
            else:
                a = 0

            output[i] = a

            e = t[i] - a

            if(e != 0):
                flag[i] = 1
            else:
                flag[i] = 0

            w = w + e*numpy.transpose(vector)
            b = b + e
            i = i + 1

        print(output)
        print(w)
        print(b)

    j = 0
    while j < len(output):
        if(output[j] == 1):
            class1[0].append(p[0,j])
            class1[1].append(p[1,j])
        else:
            class2[0].append(p[0,j])
            class2[1].append(p[1,j])
        j = j + 1

    plt.scatter(class1[0], class1[1], label = "Class 1", color = "green")
    plt.scatter(class2[0], class2[1], label = "Class 2", color = "red")
    m = -(b / w[1])/(b / w[0])
    y = m*0 + (-b/w[1])
    plt.axline((0, y), slope = m, label = "Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()

perceptron(p,t)