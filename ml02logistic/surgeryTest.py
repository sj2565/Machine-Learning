import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 

filename = 'surgeryTest.csv'
data = np.loadtxt(filename, delimiter=',')

print('type(data) :', type(data))
print('data.shape :', data.shape)

table_col = data.shape[1]

y_column = 1 
x_column = table_col - 1

# 환자의 기록과 수술 결과를 x와 y로 구분하여 저장합니다.
x = data[:, 0:x_column]
y = data[:, x_column:(x_column+1)]

print('x.shape :', x.shape)
print('y.shape :', y.shape)

seed = 0
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, random_state=seed)

model = Sequential()

model.add(Dense(units=30, input_dim=x_column, activation='relu'))

# 마지막 add는 output layer가 됩니다.
# 이 문제는 로지스틱 문제이므로 output layer는 sigmoid가 되어야 합니다.
model.add(Dense(units=y_column, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', \
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=0) 

# 키워드 검색 : "About Keras models"
print('model.inputs')
print(model.inputs)

print('model.outputs')
print(model.outputs)

print('model.layers')
print(model.layers)

# 결과를 출력합니다.
print(model.metrics_names)

score = model.evaluate(x_train, y_train)
print('train loss : %.4f' % (score[0]))
print('train accuracy : %.4f' % (score[1]))

print('-' * 50)
pred = np.argmax(model.predict(x_test), axis=-1)

for idx in range(len(pred)) :
    label = y_test[ idx ]
    # print('real : %f, prediction : %f' % (label, pred[idx]))

print('산술 연산으로 정확도를 구합니다.')
accuracy = np.sum(pred==y_test) / len(pred)
print('정확도 : %.4f' % (accuracy))

print('evaluate 함수를 사용하면 손실과 정확도를 구할 수 있습니다')
score = model.evaluate(x_test, y_test)
print('test loss : %.4f' % (score[0]))
print('test accuracy : %.4f' % (score[1]))

print('-' * 50)
print('finished')