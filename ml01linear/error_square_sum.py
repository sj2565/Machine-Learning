import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='MALGUN GOTHIC')

def myfunction(x):
    return 5.45 * x + 7.7

x = np.array([26, 28, 30, 32])
y = np.array([148, 164, 168, 183])
answer = myfunction(x)

print('x=', x)
print('y=', y)
print('answer=', answer)

print('오차 제곱의 총합 :', np.sum((answer-y)**2))
plt.plot(x, y, marker='o', color='g', linestyle='none', label='졸려')
plt.plot(x, answer, marker='o', color='r', linestyle='solid', label='잘까')

for idx in range(len(x)):
    xdata = []
    ydata = []

    xdata.append(x[idx])
    ydata.append(y[idx])

    xdata.append(x[idx])
    ydata.append(answer[idx])

    plt.plot(xdata, ydata, marker='', color='b', linestyle='solid')

filename = 'error_square_sum.png'
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(filename)