import numpy as np
from tensorflow.python.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 각 클래스의 품목 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
plt.grid(False)
filename = 'fashionMnist_01.png'
plt.savefig(filename)
print(filename + '파일이 저장됨')

# 데이터에 대한 정규화를 수행한다.
x_train, x_test = x_train/255.0, x_test/255.0

num_rows = 5    # 행수
num_cols = 5    # 열수
num_images = num_rows * num_cols    # 그리고자 하는 총 이미지 개수
plt.figure(figsize=(10, 10))

for i in range(num_images):
    # num_raws 행 num_cols 열의 ( i + 1) 번째 cell을 말한다.
    plt.subplot(num_rows, num_cols, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
# end for

filename = 'fashionMnist_02.png'
plt.savefig(filename)
print(filename + '파일이 저장됨')

# 케라스 모델을 구성하고, 학습을 진행한다.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten

nb_clasees = 10 # 이미지 10가지 종류

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units=128, activation='relu'),
    Dense(units=128, activation='relu'),
    Dense(units=nb_clasees, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=0)

# 모델에 대한 정확도 및 loss를 평가해 준다.
score = model.evaluate(x_test, y_test, verbose=2)

print('test accuracy : %.4f' % (score[1])) # 정확도
print('test loss : %.4f' % (score[0])) # 손실 함수

prediction = model.predict(x_test) # 예측하기

check_length = 5 # 예측해 볼 데이터의 개수 
print('테스트 이미지' + str(check_length) + '개 예측해보기')
print('예측 확률')
print(prediction[0:check_length])

print('예측 이미지')
print(np.argmax(prediction[0:check_length], axis = -1))

print('정답 이미지')
print(y_test[0:check_length])

# 이미지와 정답 및 예측 값에 대한 label 확률을 그려 주는 함수
def plot_image(i, prediction_array, true_label, img):
    # i : 이미지 색인 번호
    # prediction_array : 예측 확률을 저장하고 있는 배열
    # true_label : 실제 정답 데이터
    # img  : 테스트를 수행하기 위한 이미지 배열
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i], cmap=plt.cm.binary) # 이미지 그리기

    prediction_label = np.argmax(prediction_array[i]) # 예측된 값
    if prediction_label == true_label[i]:   # 맞춘 경우 파랑색으로 그리기
        mycolor = 'blue'
    else:
        mycolor = 'red'

    # 이미지 하단의 라벨 형식은 '예측된 확률%(정답) 이다.
    pred = class_names[prediction_label] # 예측값
    prob = 100.0 * np.max(prediction_array[i]) # 확률값
    answer = class_names[true_label[i]] # 정답

    plt.xlabel("{} {:6.2f}% ({})".format(pred, prob, answer), color=mycolor)

# 각각의 label에 대한 확률을 막대 그래프로 그려 주는 함수
def plot_value_array(prediction_array, true_label):
    # prediction_array : 예측 혹률을 저장하고 있는 배열
    # true_label : 실제 정답 데이터
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color='#777777')
    plt.ylim([0.0, 1.0])
    prediction_label = np.argmax(prediction_array)
    thisplot[prediction_label].set_color('red') # 예측이 틀림 (빨강)
    thisplot[true_label].set_color('blue') # 예측이 맞음 (파랑)
# end def plot_value_array

# 0번째 test용 데이터에 대하여 이미지와 확률 그래프를 그려본다.
# 예측이 맞으면 파란색으로, 틀리면 빨간색으로 데이터를 표현한다.
idx = 0 # 이미지의 색인 번호
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_image(idx, prediction, y_test, x_test)

plt.subplot(1,2,2)
plot_value_array(prediction[idx], y_test[idx])

filename = 'fashionMnist_03.png'
plt.savefig(filename)
print(filename + '파일이 저장됨')

# 반복문을 사용하여 n행 n열 데이터에 대하여 이미지와 확률 그래프를 그려본다.
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for idx in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * idx + 1)
    plot_image(idx, prediction, y_test, x_test)

    plt.subplot(num_rows, 2*num_cols, 2*idx+2)
    plot_value_array(prediction[idx], y_test[idx])

filename = 'fashionMnist_04.png'
plt.savefig(filename)
print(filename + '파일이 저장됨')