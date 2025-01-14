filename = 'wine.csv'

import pandas as pd
df_wine = pd.read_csv(filename, header=None)

print('df_wine.shape : ', df_wine.shape)
print('-'*30)

df = df_wine.sample(frac=0.15) # 15%만 샘플링
print('df.shape : ', df.shape)
print('-'*30)

data = df.values

table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column]
y = data[:, x_column:]

# 모델을 생성하고, 레이어를 추가한 다음 데이터를 학습시킨다.
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()

model.add(Dense(units=30, activation='relu', input_dim=x_column))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=y_column, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 사용할 콜백 함수에 대한 객체를 생성한다.
# ModelCheckpoint : 주어진 조건을 만족하면 해당 모델을 파일 형식으로 저장한다.
# EarlyStopping : 학습에 대한 개선의 기미가 보이지 않으면 강제로 종료를 수행한다.
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

# 파일을 저장할 경로 및 이름 정의
model_dir = './model/'  # 파일이 저장될 폴더

# 폴더가 없는 경우를 대비하여 폴더 생성
import os # 운영 체재와 관련된 파이썬 모듈
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_name = model_dir + '{epoch:02d}--{val_loss:.4f}.hdf5' # 저장될 파일 형식
# save_best_only=True : 학븟을 진행하는 동안 이전보다 개선이 된 경우에만 저장.
mcp = ModelCheckpoint(filepath=model_name, monitor='val_loss', verbose=0, save_best_only=True)

# 학습 자동 중단 설정
# patience = 100 : 테스트 오차가 좋아지지 않더라도 epoch 100번 정도는 기다려 준다.
es = EarlyStopping(monitor='val_loss', patience=100)

# validation_split = 0.2 : 20%를 검증용 데이터로 사용한다.
# callbacks에는 적용할 콜백 함수에 대한 객체를 명시하면 됨.
history = model.fit(x, y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0, callbacks=[es, mcp])

val_loss = history.history['val_loss']
accuracy = history.history['accuracy']

# 데이터 시각화
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='MALGUN GOTHIC')

print('정확도는 파란색, 오차는 빨간색으로 시각화')
plt.figure()
x_len = np.arange(len(accuracy)) # x축에 그려지는 눈금 단위

plt.plot(x_len, val_loss, 'o', c='red', markersize=3)
plt.plot(x_len, accuracy, 'o', c='blue', markersize=3)

plt.legend(['val_loss', 'accuracy']) 
savefilename = 'wineCheckStop.png'
plt.savefig(savefilename)
print(savefilename + '파일 저장됨')
print('finished')