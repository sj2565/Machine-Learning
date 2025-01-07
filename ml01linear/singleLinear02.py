import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

filename = 'singleLinear02.csv'
data = np.loadtxt(filename, delimiter=',')
print(type(data)) # <class 'numpy.ndarray'>

table_col = data.shape[1]
y_column = 1
x_column = table_col - 1
x = data[:, 0:x_column]
y = data[:, x_column:]

seed = 0  # 항상 동일한 샘플이 나오게 random seed 배정
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.25, random_state=seed)

model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=300000, batch_size=1000, verbose=1)

# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') #윈도우, 구글 콜랩

plt.title('회귀선과 실제 데이터 산점도')
plt.xlabel('x')
plt.ylabel('y')
# plt.grid(True)
plt.plot(x_train, y_train, 'k.') # 실제 데이터 산점도

train_pred = model.predict(x_train)
plt.plot(x_train, train_pred, 'r') # 회귀선
plt.savefig('singleLinear02.png')
# plt.show()

print(model.get_weights())

prediction = model.predict(x_test)
print(type(prediction))

for idx in range(len(y_test)):
    label = y_test[idx]
    pred = prediction[idx]

    print('real : %f, prediction : %f' % (label, pred))
print('finished')