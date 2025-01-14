from tensorflow.keras.datasets import boston_housing

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_x.shape)

input_dim = train_x.shape[1]
print('입력 데이터의 열 개수 : ' +str(input_dim))
print('입력 데이터의 개수 : ' +str(len(train_x)))
print('출력 데이터의 개수 : ' +str(len(test_y)))
print('0번째 데이터의 정보')
print(train_x[0])
print(train_y[0])

print('finished')

# 데이터 전처리(표준화)
x_mean = train_x.mean(axis=0)
x_std = train_x.std(axis=0)
train_x -= x_mean
train_x /= x_std
test_x -= x_mean
test_x /= x_std

y_mean = train_y.mean(axis=0)
y_std = train_y.std(axis=0)
train_y -= y_mean
train_y /= y_std
test_y -= y_mean
test_y /= y_std

print('0번째 데이터의 정보(표준화 이후)')
print(train_x[0])
print(train_y[0])

# 모델 생성 시 입력 데이터 갯수인 input_shape는 정확히 맞추어 주어야 한다.
# 촐력 데이터 갯수는 정확히 맞추어 주어야 한다.
# 이 문제는 회귀이므로 숫자 1이고, 선형 회귀이므로 activation 옵션은 반드시 'linear'이어야 한다.
# activation을 명시하지 않으면 기본 값이 'linear'이다.

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(input_dim, )),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.07))
model.summary()

callback_mode = True # True이면 콜백 함수를 사용

# 모델에 대한 학습을 진행한다.
# validation_split=0.25
# 전체 데이터 중에서 25%를 별도로 떼내에서 검증하는 용도로 사용

if callback_mode == False :
    history = model.fit(train_x, train_y, epochs=40, batch_size=32,
                        validation_split=0.25)
else :
    # monitor는 모니터링 할 대상
    es = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')

    history = model.fit(train_x, train_y, epochs=40, batch_size=32,
                        validation_split=0.25, callbacks=[es])

# 데이터에 대한 시각화를 한다.
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
plt.rc('font', family='MALGUN GOTHIC')

plt.figure()
plt.plot(history.history['loss'], 'b-', label='training data loss')
plt.plot(history.history['val_loss'], 'r--', label='validation data loss')

plt.title('회귀 모델 학습 시각화')
plt.xlabel('Epoch(검증 데이터)')
plt.legend()
filename = 'bostonLinearExam.png'
plt.savefig(filename)
print(filename + '파일 저장됨')