from tensorflow.python.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

dddd = x_train / 255
print(dddd)

print(x_train[0])

print('x_train.shape :', x_train.shape)
print('x_train.dtype :', x_train.dtype)
print('x_train.ndim :', x_train.ndim)
print('len(y_train) :', len(y_train))
print('y_train :', y_train)

print('x_test.shape :', x_test.shape)
print('x_test.dtype :', x_test.dtype)
print('x_test.ndim :', x_test.ndim)
print('len(y_test) :', len(y_test))
print('y_test :', y_test)

import matplotlib.pyplot as plt

for idx in range(10):
    someimage = x_train[idx]
    plt.imshow(someimage)
    filename = 'index' + str(idx) + '(' +str(y_train[idx]) + ').png'
    plt.savefig(filename)

print('이미지 파일로 저장이 되었다.')