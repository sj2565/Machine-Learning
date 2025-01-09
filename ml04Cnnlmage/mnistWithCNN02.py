from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Dropout

from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

# (28, 28, 1)는 (image_height, image_width, image_channels)입니다.
# 흑백은 image_channels=1이고, 컬러는 image_channels=3입니다.
# Conv2D 함수의 strides=(1, 1)가 기봅 값입니다.
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

# model.summary()

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('model.fit 실행중입니다.')
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

print('model.evaluate 실행중입니다.')
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(model.metrics_names)
print('-'*30)

print('test_acc:', test_acc)
print('-'*30)

print('test_loss:', test_loss)
print('-'*30)

# ['loss', 'accuracy']
# test_acc: 0.9926999807357788
# test_loss: 0.024219535579789953