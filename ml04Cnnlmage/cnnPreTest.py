from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

model = Sequential()

# 해당 모델의 첫 번째 layer에서만 input_shape에 해당 이미지의 차원을 입력해 주어야 한다.
# filters의 크기는 2의 n제곱으로 하는 경향이 있다.

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(150, 150, 3))) # 흑백 :1, 칼라 :3
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=512, activation='relu'))

# 이항 분류를 학습한다고 가정.
# 머신 러닝의 종류에 따라, 마지막 layer의 activation 함수의 내용이 달라져야 한다.
y_column=1
model.add(Dense(units=y_column, activation='sigmoid'))

model.summary()