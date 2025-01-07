import numpy as np
filename ='multiLinear01.csv'

data = np.loadtxt(filename, delimiter=',')

print(data.shape) # 몇행 몇열인가?

table_col = data.shape[1] # data행렬에서 전체 행의 갯수와 열의 갯수를 반환 => shape[0] : 행 , shape[1] : 열
y_column = 1
x_column = table_col - y_column  # 결국 x_column은 2

# 입력 데이터와 출력(정답, label) 데이터를 분리하기
x = data[:, 0:x_column] # [여기는 row(행):, 여기는column(열)] => 행은 전부 출력하고 열은 0:2 이므로 두번째 열까지 출력
y = data[:, x_column:]

print('x', x)
print('y', y)

# 훈련용 데이터와 테스트용 데이터를 분리하기

# from sklearn.model_selection import train_test_split
#
# seed = 0  # 항상 동일한 샘플이 나오게 random seed 배정
# x_train, x_test, y_train, y_test = \
#     train_test_split(x, y, test_size=0.25, random_state=seed)
#
# print('x_train : ', x_train)
# print('-'*30)
#
# print('x_test : ', x_test)
# print('-'*30)
#
# print('y_train : ', y_train)
# print('-'*30)
#
# print('y_test : ', y_test)
# print('-'*30)
#
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense