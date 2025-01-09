import numpy as np

x = np.array([26, 28, 30, 32])
y = np.array([148, 164, 168, 183])

w = np.arange(0.0, 10.91, 0.01)
print('기울기 w')
print(w)
print('-'*30)

def myfunction(w, x):
    data = w * x + 7.7
    return np.sum((data-y)**2)

loss = [] # 오차가 저장 될 리스트
for some in w :
    data = myfunction(some, x)
    loss.append(data)

print('오차 함수 loss')
print(loss)
print('-'*30)

minlost = min(loss)
print('오차 함수 최소값')
print(minlost)
print('-'*30)

import matplotlib.pyplot as plt
plt.rc('font', family="MALGUN GOTHIC")
plt.plot(w, loss, color='g', linestyle='solid', linewidth=1, label='이차곡선')
plt.grid(True)
filename = 'compareMSE.png'
plt.savefig(filename)
print(filename + '이 저장되었습니다')
