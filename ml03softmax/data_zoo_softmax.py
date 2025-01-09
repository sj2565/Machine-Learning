menu = int(input('콜백 함수 사용 여뷰 => no(0), yes(1) : '))

filename = 'zoo.data.txt'

import pandas as pd
df = pd.read_csv(filename, index_col='name')
print(df.head())

# 상관 계수 시각화
corr = df.corr()    # 상관 계수 매트릭스

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15)) # 도화지 크게
sns.heatmap(corr, linewidth=0.1, vmax=0.5, linecolor='white', annot=True)
filename = 'data_zoo_corr_image.png'
plt.savefig(filename)
print(filename + '파일 저장됨')

'''
상관 계수는 -1.0 ~ 1.0 사이의 값을 가진다.
동물의 타입을 결정 짓기 위하여 eggs 컬럼이 양의 상관 관계를 가지고, backbone이 가장 큰 음의 상관 계수를 가진다.
상관 관계가 비교적 큰 컬럼들에 대하여 pairplot을 그려 보겠다.
'''
pair_df = df[['eggs', 'milk', 'backbone', 'venomous', 'type']].reset_index()
sns.palplot(pair_df, hue='type')
filename = 'data_zoo_pair_plot.png'
plt.savefig(filename)
print(filename + '파일 저장됨')

data = df.values

# 머신 러닝을 위한 기초 변수들
table_col = data.shape[1]
y_column = 1
x_column = table_col - y_column
x = data[:, 0:x_column]
y_raw = data[:, x_column:].ravel()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_raw)
y = le.transform(y_raw)

x = x.astype(float)
y = y.astype(float)

from sklearn.model_selection import train_test_split
seed = 1234
x_train, y_train, x_test, y_test = \
    train_test_split(x, y, test_size=0.3, random_state = seed)

from keras.utils import np_utils
nb_classes = 7
y_train=np_utils.to_categorical(y_test, num_classes=nb_classes, dtype='float32')

# print(x_train)
# print(y_train)

from tensorflow.python.keras.models import Sequential
model = Sequential()

from tensorflow.python.keras.layers import Dense
model.add(Dense(units=nb_classes, input_shape=(x_column, ), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('데이터를 훈련 중입니다. 잠시만 기다려 주세요')

if menu == 0:   # 콜백 함수 사용 안 함
    histroy = model.fit(x_train, y_train, epochs=10000, verbose=1, validation_split=0.3)
else:  # 콜백 함수를 사용한 학습 자동 중단하기
    from keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=500)
    model.fit(x_train, y_train, epochs=10000, verbose=1, validation_split=0.3, callbacks=[es])

print(histroy)

# 정확도와 손실 함수에 대한 그래프를 그려 본다.
import matplotlib.pyplot as plt
plt.rc('font', family='MALGUN GOTHIC')

accuracy = histroy.histroy['accuracy']
val_accuracy = histroy.histroy['val_accuracy']

plt.figure()

plt.plot(accuracy, 'b--', label='training accuracy')
plt.plot(val_accuracy, 'r--', label='validation accuracy')

plt.title('epoch에 따른 정확도 그래프')
plt.legend()
if menu == 0:
    filename = 'data_zoo_figure_01(no).png'
else :
    filename = 'data_zoo_figure_01(yes).png'

plt.savefig(filename)
print(filename + ' 파일 저장됨')
# -------------------------------------------------------------------
loss = histroy.histroy['loss']
val_loss = histroy.histroy['val_loss']

plt.figure()

plt.plot(loss, 'b--', label='training loss')
plt.plot(val_loss, 'r--', label='validation loss')

plt.title('epoch에 따른 정확도 그래프')
plt.legend()
if menu == 0:
    filename = 'data_zoo_figure_02(no).png'
else:
    filename = 'data_zoo_figure_02(yes).png'

plt.savefig(filename)
print(filename + ' 파일 저장됨')

# 예측값, 정답, 확률 값 0 ~ 6 까지를 csv파일 형식으로 저장해 본다.
totallist = []   # csv 파일로 저장될 리스트
hit = 0.0 # 데이터를 맞춘 개수

import numpy as np

for idx in range(len(x_test)):
    H = model.predict(np.array([x_test[idx]]))
    prediction = np.argmax(H, axis= -1)

    sublist=[]
    sublist.append(prediction[0]) # 예측값
    sublist.append(int(y_test[idx]))    # 정답
    
    _H=H.flatten() # 1차원으로 변경하기
    for jdx in range(len(_H)):
      sublist.append(_H[jdx])
    # inner for

    totallist.append(sublist)

    hit += float(prediction[0] == int(y_test[idx]))
# outer for

hitrate=hit/len(x_test)
print('정확도 : %.4f' % (100*hitrate))

colnames = ['예측값', '정답', '확률값0', '확률값1', '확률값2', '확률값3', '확률값4', '확률값5', '확률값6']
df = pd.DataFrame(totallist, columns=colnames)
csvfile = 'data_zoo_excel_csv.csv'
df.to_csv(csvfile, index=False, encoding='cp949')
print(filename + ' 파일 저장됨')