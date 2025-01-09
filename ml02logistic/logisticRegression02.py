'''
타이타닉 호에서 생존과 죽음 정보에 대한 로지스틱 회귀를 실습해 봅니다.
성별 컬럼을 수치 데이터로 코딩 변경합니다.
나이에 대한 결측치 값들을 평균 값으로 대체합니다.
몇 등석(pclass 컬럼)에 탑승했는 지 정보를 이용하여 일등석/이등석에 대한 더미 코딩을 수행합니다.
성별과 나이와 일등석 및 이등석 컬럼을 독립 변수로 사용합니다.
종속 변수는 생존 여부를 저장하고 있는 컬럼입니다.
데이터를 분리하고, 정규화를 수행합니다.
로지스틱 모델을 생성하고, 예측을 수행해 봅니다.
독립 변수들의 가중치 정보를 시각화합니다.
샘플 데이터로 예측을 수행해 봅니다.
혼돈 매트릭스, 정확도, 클래스에 대한 보고서 등을 출력해 봅니다.
'''
import pandas as pd

filename = 'titanic.csv'
data = pd.read_csv(filename)
print(data.shape)

# survived 생사 여부(1이 생존)
print(data.columns)

# 코딩 변경
data['sex'] = data['sex'].map({'female':1,'male':0})

# 결측치 처리
data['age'].fillna(value=data['age'].mean(), inplace=True)


# feature 분리하기
# Pcalss의 경우 1등석에 탔는지, 2등석에 탔는지에 대해 각각의 feature로 만들어주기 위해 컬럼을 새로 생성해보자.
data['firstclass'] = data['pclass'].apply(lambda x: 1 if x == 1 else 0)
data['secondclass'] = data['pclass'].apply(lambda x: 1 if x == 2 else 0)

concern=['sex', 'age', 'firstclass', 'secondclass'] # 관심 컬럼
x_data = data[concern]
y_data = data['survived']

# 입력과 출력 데이터 셋을 다음과 같이 분리합니다.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# 로지스틱 회귀) regularation을 사용하기 때문에 정규화를 해주는 것을 권장합니다.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# 그래서 위 코드에서도 학습세트로 fit을 한 번 해주었기 때문에, 평가세트에서는 별도로 fit을 할 필요 없이 바로 transform하면 되는 거다.
x_test = scaler.transform(x_test)

# 모델 생성하기
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
print('# train 정확도 : %.3f' % (train_score))

test_score = model.score(x_test, y_test)
print('# test 정확도 : %.3f' % (test_score))

import numpy as np

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ', model.coef_)
print('절편 : ', model.intercept_)

# sex, age, firstclass, secondclass 순으로 계수가 나왔습니다
# 성별과 일등석 탑승 여부의 계수가 크게 나오는 데, 이 두 컬럼이 생사를 결정하는 데
# 많은 영향을 준다고 볼수 있습니다.
# 반면 나이 계수는 음수가 나오는데 나이가 많을수록 생존 확률이 낮아짐을 의미합니다.
import matplotlib
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from pandas import Series
myseries=Series(np.reshape(model.coef_, -1))
myseries.plot(kind='bar')
filename = 'logisticRegression02_01.png'
plt.title('독립 변수들의 가중치')
plt.xticks(np.arange(len(concern)), concern, rotation ='horizontal')
plt.savefig(filename, dpi=400, bbox_inches='tight' )
print(filename + ' 파일이 저장되었습니다.')

# 샘플 데이터로 예측하기
import numpy as np
soo = np.array([0.0, 20.0, 0.0, 0.0])
hee = np.array([1.0, 17.0, 1.0, 0.0])
minho = np.array([0.0, 32.0, 1.0, 0.0])
sample = np.array([soo, hee, minho])

sample = scaler.transform(sample) # 데이터 정규화

print(model.predict(sample))
# [0 1 0]
# 두 번재 사람인 hee만 살아 남습니다.

# 확률을 확인합니다.
print(model.predict_proba(sample))

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
filename = 'logisticRegression02_02.png'
plt.savefig(filename, dpi=400, bbox_inches='tight' )
print(filename + ' 파일이 저장되었습니다.')
print('finished')