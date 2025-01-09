import pandas as pd

filename = 'iris.csv'
data = pd.read_csv(filename)
print(data.shape)

print(data.columns)

x_data = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y_data = data['Name']

# 입력과 출력 데이터 셋을 다음과 같이 분리합니다.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

# 로지스틱 회귀) regularation을 사용하기 때문에 정규화를 해주는 것을 권장합니다.
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
# Sex, Age, FirstClass, SecondClass 순으로 계수가 나왔습니다
# 성별과 일등석 탑승 여부의 계수가 크게 나오는 데, 이 두 커럼이 생사를 결정하는 데
# 많은 영향을 준다고 볼수 있습니다.
# 반면 나이 계수는 음수가 나오는데 나이가 많을수록 생존 확률이 낮아짐을 의미합니다.

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
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

# plt.figure()
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap='YlGnBu', fmt='g')
# plt.tight_layout()
plt.title("Confusion matrix", y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predict label")
# plt.show()
filename = 'logisticRegression01_01.png'
plt.savefig( filename, dpi=400, bbox_inches='tight' )
print( filename + ' 파일이 저장되었습니다.')
print('finished')