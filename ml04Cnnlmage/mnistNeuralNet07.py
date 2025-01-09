from tensorflow.python.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_column = 28 * 28
# CNN은 데이터를 4차원 형식으로 만들어 줘야 한다.
x_train=x_train.reshape((60000, 28, 28, 1)) # 형상 변경
x_train=x_train.astype(float)/255

x_test=x_test.reshape((10000, 28, 28, 1))
x_test=x_test.astype(float)/255

print('before y_train[0]')
print(y_train[0])

from keras.utils import np_utils

# label에 대한 onehot encoding을 적용해 준다.
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print('after y_train[0] :', y_train[0])

#  모델을 생성하고, 학습을 진행한다.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dense, Dropout

model = Sequential()

nb_classes = 10 # 숫자 0 ~ 9 까지

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# hidden layer 01
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))

# hidden layer 02
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(units=nb_classes, activation='softmax'))

# 1번 테스트
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 2번 테스트
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('model.fit() 실행중입니다.')
history = model.fit(x_train, y_train, validation_split=0.3, epochs=5, batch_size=64, verbose=1)

print('history의 모든 데이터 목록 보기')
print(history.history.keys())
#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
# compile 함수 사용 시 metrics를 명시하지 않으면, accuracy는 보여 주지 않는다.

print('model.evaluate() 실행중입니다.')
score = model.evaluate(x_test, y_test, verbose=1)

print('지표 목룍')
print(model.metrics_names)

print('test loss : %.4f' %(score[0]))
print('test acc : %.4f' %(score[1]))

import matplotlib.pyplot as plt
plt.rc('font', family='MALGUN GOTHIC')

# 정확도에 대하여 그래프를 그려 보자.
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
filename='mnistNeuralNet01.png'
plt.savefig(filename)
print(filename + '파일 저장됨')

# 손실 함수에 대하여 그래프를 그려 보자.
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
filename='mnistNeuralNet02.png'
plt.savefig(filename)
print(filename + '파일 저장됨')