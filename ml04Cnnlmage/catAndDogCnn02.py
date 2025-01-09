# 이미지를 읽어 들일 폴더를 지정한다.
import os
target_folder = '../datasets/cats_and_dogs_small'
train_folder = os.path.join(target_folder, 'train') # 훈련용 이미지 폴더
validation_folder = os.path.join(target_folder, 'validation') # 검증용 이미지 폴더

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator 객체를 생성한다.
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 읽어 들일 이미지의 너비와 높이에 관련된 변수 (target_size)
TARGET_WIDTH, TARGET_HEIGHT = 150, 150
BATCH_SIZE = 20

# 케라스 모델에서 사용될 제너레이터 객체를 생성한다.
# batch_size : 20, 이항분류이므로 class_mode : binary
train_generator=train_datagen.flow_from_directory(
    directory=target_folder,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary')

'''
Found 2000 images belonging to 2 classes.
해당 폴더에 2부류의 데이터 이미지 2000개가 발견이 되었다.
'''

validation_generator=validation_datagen.flow_from_directory(
    directory=validation_folder,
    target_size=(TARGET_WIDTH, TARGET_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary')

# generator 객체
for data_batch, label_batch in train_generator:
    # (배치 사이즈, TARGET_WIDTH, TARGET_HEIGHT, color_mode)
    # color_mode가 3이므로 컬러 이미지이다.
    print('배치 데이터 크기 : ', data_batch.shape)
    print('배치 레이블(정답) 크기 : ', label_batch.shape)
    break

# 케라스 모델을 생성한다.
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping

model = Sequential()

COLOR_MODE = 3 # 컬러 이미지
# 첫 번째 layer에 이미지에 대한 입력 정보를 넣어 주어야 한다.
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=(TARGET_WIDTH, TARGET_HEIGHT, COLOR_MODE)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 컨볼루션/맥스 풀링은 이미지를 2차원 형태로 다루는 데, 최종 Dense 레이어에는 1차원이 필요하다.
# Flatten 함수를 이용해서 저차원으로 변경해 준다
model.add(Flatten())

model.add(Dense(units=512, activation='relu'))

# 마지막 layer에는 해당 회귀/분류 알고리즘에 맞도록 지정해주어야 한다.
# 강아지와 고양이에 대한 이항 분류 문제이다.
model.add(Dense(units=1, activation='sigmoid'))

# 해당 모델의 layer에 대한 개략적인 정보를 확인한다.
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es=EarlyStopping(patience=3, monitor='val_loss')

history = model.fit(
    x=train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=1,
    callbacks=[es])

print('다음 실습을 위하여 모델을 파일 형태로 저장한다.')
model.save('cats_and_dogs_small.h5')

# 학습 결과에 대한 데이터 시각화를 진행한다.
# 훈련용과 검증용에 대한 모델의 손실 함수와 정확도
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt

x = range(len(accuracy))

plt.figure()
plt.plot(x, accuracy, 'bo', label='training accuracy')
plt.plot(x, val_accuracy, 'b', label='validation accuracy')
plt.title('training and validation accuracy')
plt.legend(loc = 'best')    # 범례는 가장 적절한 위치에...
filename = 'catAndDog02_01.png'
plt.savefig()
print(filename + '파일 저장됨')

plt.figure()
plt.plot(x, loss, 'bo', label='training loss')
plt.plot(x, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend(loc = 'best')    # 범례는 가장 적절한 위치에...
filename = 'catAndDog02_02.png'
plt.savefig()
print(filename + '파일 저장됨')