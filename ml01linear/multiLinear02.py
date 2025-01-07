import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 관련 공식) y=3*x1+2*x2+1*x3-4

filename = 'multiLinear02.csv'
data = np.loadtxt(filename, delimiter=',')

table_col = data.shape[1]
y_column = 1
x_column = table_col - 1
x = data[:, 0:x_column]
y = data[:, x_column:]

seed = 0
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.30, random_state=seed)

model = Sequential()

model.add(Dense(units=y_column, input_dim=x_column, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=100000, batch_size=100, verbose=0)

print(model.get_weights())

prediction = model.predict(x_test)

for idx in range(len(y_test)):
    label = y_test[idx]
    pred = prediction[idx]

    print('real : %f, prediction : %f' % (label, pred))

print('finished')

# epochs=100000, batch_size=100
# [array([[2.4673119],
#        [2.1066527],
#        [0.5739649]], dtype=float32), array([1.8578396], dtype=float32)]
# real : 24.000000, prediction : 24.000000
# real : 48.000000, prediction : 48.000000
# real : 32.000000, prediction : 31.999998
# finished

# [array([[3.2239995 ],
#        [1.4242057 ],
#        [0.64820504]], dtype=float32), array([1.500924], dtype=float32)]
# real : 24.000000, prediction : 24.000000
# real : 48.000000, prediction : 48.000000
# real : 32.000000, prediction : 31.999998
# finished