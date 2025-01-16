import tensorflow as tf
import tensorflow.keras.datasets as ds
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist

# MNIST 읽고 텐서 모양 출력
#(x_train, y_train),(x_test, y_test)=ds.mnist.load_data()
(x_train, y_train), (x_test, y_test) = load_mnist()
x_train=x_train.reshape(60000,28,28)
yy_train=tf.one_hot(y_train,10,dtype=tf.int8) # 원핫 코드로 변환
print("MNIST: ",x_train.shape,y_train.shape,yy_train.shape)

