'''
단항 선형 회귀 문제를 분석해 봅니다.
훈련용 파일과 점검용 파일을 별도로 가지고 있습니다.
모델을 학습한 다음 데이터 들과 회귀 선을 이용하여 시각화를 수행합니다.
결정 계수를 산술 공식에 의하여 풀어 보고, scikit에서 제공하는 score 함수를 이용하여 풀어 봅니다.
차후에 재사용하기 위하여 모델을 저장해 봅니다.
'''
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

import numpy as np
filename = 'linearTest01.csv'
training = np.loadtxt(filename, delimiter=',', skiprows=1)

x_column = training.shape[1] - 1
print('x_column : ' + str(x_column))

x_train = training[:, 0:x_column] # 학습용 입력 데이터 
y_train = training[:, x_column:] # 정답 label

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ', model.coef_) # W값
print('절편 : ', model.intercept_) # b값
print('잔차의 제곱합(Cost) : ', model._residues) #

# 시각화
plt.title('그래프')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.plot(x_train, y_train, 'k.') # 실제 데이터 산점도

train_pred = model.predict(x_train)
plt.plot(x_train, train_pred, 'r') # 회귀선
filename='linearRegression01_01.png'
plt.savefig(filename)
# plt.show()
print(filename + ' 파일 저장됨')

# RSS = (정답 - 예측치) 제곱의 총합
print('RSS : %.3f' % (np.sum((y_train - train_pred)**2)))

# 테스트용 데이터 셋
filename = 'linearTest02.csv'
testing = np.loadtxt(filename, delimiter=',', skiprows=1)

x_column = testing.shape[1] - 1

x_test = testing[:, 0:x_column]
y_test = testing[:, x_column:]
# print('x_test :', x_test)
# print('y_test :', y_test)

# 산술 공식에 의한 결정 계수 구하기
y_test_mean = np.mean(np.ravel(y_test))

# TSS) 편차의 제곱의 합
TSS = np.sum((np.ravel(y_test)-y_test_mean)**2)
print('TSS : %.3f' % (TSS))

# RSS) 회귀 식과 평균 값의 차이
RSS = np.sum((np.ravel(y_test)-np.ravel(model.predict(x_test)))**2)
print('RSS : %.3f' % (RSS))

R_Squared = 1 - (RSS/TSS) # 결정 계수
print('R_Squared : %.3f' % (R_Squared))

print('score 함수는 결정 계수를 구해주는 함수입니다.')
print('R_Squared : %.3f' % (model.score(x_test, y_test)))

# 모델 파일로 저장하기
import pickle

filename = 'linearModel.sav'

pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(x_test, y_test)

print('loaded_model R_Squared : %.3f' % (result))

# 사이킷 런의 joblib 모듈 사용하기
import joblib

filename = 'linearModel2.sav'

joblib.dump(model, filename)

job_model = joblib.load(filename)

job_result = job_model.score(x_test, y_test)

print('job_model R_Squared : %.3f' % (job_result))

print('finished')