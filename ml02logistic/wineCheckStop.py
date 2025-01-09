import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

filename = 'wine.csv'
df_wine = pd.read_csv( filename, header= None )

print('df_wine.shape :', df_wine.shape)
print('-' * 40)

df = df_wine.sample(frac = 0.15)
data = df.values
print('df.shape :',  df.shape)
print('-' * 40)

table_col = data.shape[1]

y_column = 1 
x_column = table_col - y_column  

x = data[:, 0:x_column ]
y = data[:, x_column:(x_column+1) ]

model = Sequential()
model.add(Dense(30,  input_dim=x_column, activation='relu')) # 입력 12개
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 만들기
model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_name="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장
mcp = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=0, save_best_only=True)

# 학습 자동 중단 설정
es = EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(x, y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0, callbacks=[es,mcp])

# y_vloss에 테스트셋으로 실험 결과의 오차 값을 저장
val_loss=history.history['val_loss']

print(model.metrics_names)

# 학습 셋으로 측정한 정확도의 값을 저장합니다.
accuracy=history.history['accuracy']

# x값을 지정하고 정확도를 파란색으로, 오차를 빨간색으로 표시
x_len = np.arange(len(accuracy))
plt.plot(x_len, val_loss, "o", c="red", markersize=3)
plt.plot(x_len, accuracy, "o", c="blue", markersize=3)
plt.legend(['val_loss', 'accuracy'])
plt.savefig('wineCheckStop.png')
# plt.show()
print('finished')
