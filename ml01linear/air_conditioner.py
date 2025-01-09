import numpy as np

x=[26, 28, 30, 32]
y=[148, 164, 168, 183]

mx = np.mean(x)
my = np.mean(y)
print('x의 평균 값:', mx)
print('y의 평균 값:', my)

divisor = sum([(mx-i)**2 for i in x])
print('분모:', divisor)

def top(x, mx, y, my):
    d = 0

    for i in range(len(x)):
        d += (x[i]-mx)*(y[i]-my)
    return d

dividend = top(x, mx, y, my)
print('분자:', dividend)

w = dividend / divisor
b = my - (mx*w)
print('기울기 w :', w)
print('절편 b :', b)