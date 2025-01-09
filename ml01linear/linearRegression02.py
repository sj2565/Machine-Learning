'''
다중 선형 회귀 문제를 분석해 봅니다.
컬럼 정보를 출력합니다.
훈련과 테스트를 위하여 데이터를 8대2로 분류합니다.
'''
#
#
#

# plt.rc('font', family='Malgun Gothic')

import pandas as pd

# 가격(rent) 컬럼이 종속 변수입니다.
data = pd.read_csv("manhattan.csv")
print('파일의 컬럼 정보')
print(data.columns)

# 입력과 출력 데이터 셋을 다음과 같이 분리합니다.
x = data[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = data[['rent']]

from sklearn.model_selection import train_test_split

# SEED = 1234
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=SEED)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

# 데이터를 시각화해 봅니다.
y_predict = model.predict(x_test)

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.figure()
plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("정답 데이터")
plt.ylabel("예측 데이터")
plt.title("다중 선형 회귀")
plt.xlim([0, 20000])
plt.ylim([0, 20000])
filename = 'linearRegression02_01.png'
plt.savefig(filename, dpi=400, bbox_inches='tight' )
print(filename + ' 파일이 저장되었습니다.')

print('학습(fit) 이후에 회귀 계수 확인하기')
print('기울기 : ', model.coef_)
print('절편 : ', model.intercept_)

# 상관(관련)성을 산점도로 시각화해봅니다.
# 주택의 면적('size_sqft')과 가격 'rent'의 시각화
plt.figure()
plt.scatter(data[['size_sqft']], data[['rent']], alpha=0.4)
plt.xlabel('size_sqft')
plt.ylabel('rent')
plt.xlim([0, 20000])
plt.ylim([0, 20000])
filename = 'linearRegression02_02.png'
plt.savefig(filename, dpi=400, bbox_inches='tight' )
print(filename + ' 파일이 저장되었습니다.')

prediction = model.predict(x_test)
# print(prediction)

print('score 함수는 결정 계수를 구해주는 함수입니다.')
print('R_Squared : %.3f' % (model.score(x_test, y_test)))

print('finished')