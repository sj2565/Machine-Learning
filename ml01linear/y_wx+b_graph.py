import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='MALGUN GOTHIC')

def myFunction(x):
    return 5.45 * x + 7.7

x = np.arange(26.0, 32.1, 2.0)
y = myFunction(x)

print('x=', x)
print('y=', y)

y_answer = [148.0, 164.0, 168.0, 183.0]
print('y_answer=', y_answer)

plt.plot(x, y_answer, marker='o', color='g', linestyle='none', label='label')

plt.plot(x, y, marker='', color='r', linestyle="solid", label="그려보자")

plt.xlim(x.min()-2, x.max()+2)
plt.ylim(y.min()-10, y.max()+10)
plt.legend(loc='upper left')
plt.grid(True)
filename = 'figure01.png'
plt.savefig(filename)
plt.show()