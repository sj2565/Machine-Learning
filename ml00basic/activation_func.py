import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

import numpy as np

x = np.arange(-5, 5, 0.01)
sigmoid_x = [sigmoid(z) for z in x]
tanh_x = [math.tanh(z) for z in x]
relu = [0 if z < 0 else z for z in x]


import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

plt.figure()

plt.axhline(0, color='gray')
plt.axvline(0, color='gray')

plt.plot(x, sigmoid_x, 'b--', label='sigmoid')
plt.plot(x, tanh_x, 'r--', label='tanh')
plt.plot(x, relu, 'g--', label='relu')
plt.ylim([-1.5, 2])
plt.legend()
filename='activation_func.py.png'
plt.savefig(filename)
# # plt.show()
print(filename + ' 파일 저장됨')