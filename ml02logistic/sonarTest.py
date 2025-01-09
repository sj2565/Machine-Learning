import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.models import load_model

filename= 'sonarTest.csv'
df=pd.read_csv(filename, header=None)

# print(df.head())
# print('-' * 50)

# print(df.info())
# print('-' * 50)

data=df.values

table_col=data.shape[1]

y_column=1
x_column=table_col - 1

# print(x_column)

x=data[:, 0:x_column]
y_imsi=data[:, x_column]

# print('-' * 50)
# print(y_imsi)

e = LabelEncoder()
e.fit(y_imsi)
y = e.transform(y_imsi)

# 다음 오류를 막기 위하여 실수형 타입으로 변경해야 합니다.
# ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float).
x = x.astype(np.float)
y = y.astype(np.float)

# print('-' * 50)
# print(y)

seed=0
x_train, x_test, y_train, y_test=\
    train_test_split(x, y, test_size=0.3, random_state=seed)

model=Sequential()

model.add(Dense(units=24, input_dim=x_column, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=y_column, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=0)

print(' # 모델을 컴퓨터에 저장합니다.')
model.save('my_model.h5') 

# del model # 테스트를 위해 메모리 내의 모델을 삭제

# print(' # 모델을 다시 읽어 들입니다.')
# model=load_model('my_model.h5')

# 불러온 모델로 테스트를 다시 실행한다.
print(model.metrics_names)

score = model.evaluate(x_test, y_test)
print('test loss : %.4f' % (score[0]))
print('test accuracy : %.4f' % (score[1]))
print('-' * 40)

print('finished')