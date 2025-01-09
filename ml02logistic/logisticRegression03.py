import pandas as pd

filename = 'pima-indians-diabetes.csv'
data = pd.read_csv(filename)
print(data.shape)

print(data.columns)

concern=['pregnant', 'plasma', 'pressure', 'thickness', 'insulin', 'bmi', 'pedigree', 'age']
x_data = data[concern]
y_data = data['class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

# 모델 생성하기
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
print('# train 정확도 : %.3f' % (train_score))

test_score = model.score(x_test, y_test)
print('# test 정확도 : %.3f' % (test_score))

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ', model.coef_)
print('절편 : ', model.intercept_)

import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from pandas import Series
import numpy as np

myseries=Series(np.reshape(model.coef_, -1))
myseries.plot(kind='bar')
filename = 'logisticRegression03_01.png'
plt.title('독립 변수들의 가중치')
plt.grid(True)
plt.xticks(np.arange(len(concern)), concern, rotation ='horizontal')
plt.savefig(filename, dpi=400, bbox_inches='tight' )
print(filename + ' 파일이 저장되었습니다.')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print('test results:')

test_pred = model.predict(x_test)
print('confusion matrix:')
cf_matrix = confusion_matrix(y_test, test_pred)
print(cf_matrix)
accuracy = accuracy_score(y_test, test_pred)
print('\n정확도 : %.3f' % (100 * accuracy))
print('\nclassification report:')
cl_report = classification_report(y_test, test_pred)
print(cl_report)
print('-' * 40)

# 히트맨 생성
import seaborn as sns
plt.figure()
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap='YlGnBu', fmt='g')
# plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predict label")
# plt.show()
filename = 'logisticRegression03_02.png'
plt.savefig( filename, dpi=400, bbox_inches='tight' )
print( filename + ' 파일이 저장되었습니다.')
print('finished')