# 데이터 불러오기
from tensorflow.keras.datasets import boston_housing
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

input_dim=train_X.shape[1]
print('입력 데이터 열 개수 : ' + str(input_dim))
print('입력 데이터의 개수 : ' + str(len(train_X)))
print('출력 데이터의 개수 : ' + str(len(test_X)))
print('0번째 데이터 정보(표준화 이전)')
print(train_X[0])
print(train_Y[0])

# 데이터 전처리(표준화)
x_mean = train_X.mean(axis=0)
x_std = train_X.std(axis=0)
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean(axis=0)
y_std = train_Y.std(axis=0)
train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

print('0번째 데이터 정보(표준화 이후)')
print(train_X[0])
print(train_Y[0])

import tensorflow as tf

# True 또는 False의 값을 가질 수 있습니다.
callback_mode=False
# callback_mode=True

# Dataset(입력 : 13, 출력 : 1)에 대한 회귀 모델을 생성합니다.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='mse')
model.summary()

# 회귀 모델에 대한 학습을 진행합니다.
if callback_mode==False:
    history = model.fit(train_X, train_Y, epochs=40, batch_size=32, validation_split=0.25, verbose=0)
else:
    es=tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    history = model.fit(train_X, train_Y, epochs=40, batch_size=32,
                    validation_split=0.25, verbose=0, callbacks=[es])

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Malgun Gothic')

plt.figure()
plt.plot(history.history['loss'], 'b-', label='training data loss')
plt.plot(history.history['val_loss'], 'r--', label='validation data loss')
plt.title('회귀 모델 학습 결과 시각화')
plt.xlabel('Epoch(검증 데이터)')
plt.legend()
filename= 'boston01.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')
# plt.show()

# 회귀 모델에 대한 평가를 수행합니다.
model.evaluate(test_X, test_Y)

pred_Y = model.predict(test_X)

plt.figure(figsize=(5,5))
plt.plot(test_Y, pred_Y, 'b.')
plt.axis([min(test_Y), max(test_Y), min(test_Y), max(test_Y)])

# y=x에 해당하는 대각선
plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls="--", c=".3")
plt.xlabel('real data')
plt.ylabel('prediction data')
plt.title('실제 가격 vs 예측 가격')
filename= 'boston02.png'
plt.savefig(filename)
print(filename + ' 파일 저장됨')
# plt.show()
print('finished')

