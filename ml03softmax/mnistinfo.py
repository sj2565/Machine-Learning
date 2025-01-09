from tensorflow.python.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train.shape', x_train.shape)   # 행렬
print('x_train.ndim', x_train.ndim) # 차원
print('x_train.dtype', x_train.dtype)   # 데이터 타입
print('len(y_train.shape)', len(y_train))
print('y_train', y_train)

print('x_test.shape', x_test.shape)   # 행렬
print('x_test.ndim', x_test.ndim) # 차원
print('x_test.dtype', x_test.dtype)   # 데이터 타입
print('len(y_test.shape)', len(y_test))
print('y_test', y_test)