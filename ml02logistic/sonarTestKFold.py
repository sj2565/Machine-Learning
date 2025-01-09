import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense

filename = 'sonarTest.csv'
df = pd.read_csv(filename, header=None)

data = df.values 
table_col = data.shape[1]

y_column = 1
x_column = table_col - y_column

x = data[:, 0:x_column ].astype(np.float)
y_obj = data[:, x_column:]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj).astype(np.float)

seed = 0 # 랜덤 시드 값
n_fold = 30 # 겹의 갯수
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

cost = []
accuracy = []

print('반복문을 실행 중입니다.')
for train, test in skf.split(x, y):
    model = Sequential()
    model.add(Dense(units=24, input_dim=x_column, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x[train], y[train], epochs=200, batch_size=5, verbose=0)

    score = model.evaluate(x[test], y[test], verbose=0)
    
    cost.append( score[0] )
    accuracy.append( score[1] )
# end for

print('-' * 40)
print('손실 함수 :')
print(cost)
print('평균 손실 : %.3f' % (sum(cost)/len(cost)))
print('-' * 40)

# K=10이므로 10개의 정확도이고, 이것의 평균을 구하면 됩니다.
print('정확도 :')
print(accuracy)
print('평균 정확도 : %.3f' % (sum(accuracy)/len(accuracy)))
print('-' * 40)