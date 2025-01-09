from tensorflow.python.keras.applications.vgg16 import VGG16

model = VGG16()
print('type(model)')
print(type(model))

model.summary()

# VGG16 모델은 입력 이미지의 크기가 224*224이다.

target_width, target_height = 224, 224
img_target = '../image/'    # 이미지 출처
img_source = '../myimage/'  # 파일을 저장할 이미지 경로

# 이미지가 많으면 os 모듈의 listdir 메소드를 사용하는게 좋다.
mylist = ['mydog.png', 'cat.jpg', 'myrabbit.jpg', 'fox.jpg']

from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
plt.rc('font', family = 'MALGUN GOTHIC')

# 예측을 수행하고자 하는 전처리된 이미지 리스트
total_data =[]

for imgname in mylist:
    img=load_img(img_target + imgname, target_size=(target_width, target_height))
    #print(type(img))
    #print(img)
    plt.imshow(img)

    # 읽어 들인 파일을 다시 파일로 지정해 본다.
    filename=img_source + '사전 학습 모델' + imgname
    plt.savefig(filename)
    #print(filename + '파일로 저장되었다')

    # 해당 이미지를 배열 형태(numpy.ndarray)로 변환한다.
    img_arr = img_to_array(img)
    # print(img_arr)

    # preprocess_input 함수는 이미지의 픽셀 평균을 0으로 변환시키는 함수다.
    # 해당 이미지를 VGG16 모델의 학습 환경과 동일하게 형상을 변경해 준다.
    pre_input=preprocess_input(img_arr)
    #print(pre_input)
    print('pre_input.shape : ', pre_input.shape)

    total_data.append(pre_input)
    print('*'*30)
# end for

# 실습에 사용할 모든 이미지를 합쳐서 새로운 배열에 저장한다.
import numpy as np
arr_input = np.stack(total_data)

print('입력 데이터의 형상을 확인한다')
# (4, 224, 224, 3)의 숫자 4는 이미지 개수를 의미한다.
print('shape of arr input : ', arr_input.shape)
#print(arr_input)

# 해당 배열(arr_input)을 모델에 넣어서 예측을 수행해 본다.
H = model.predict(arr_input)

# VGG16 모델의 output layer는 class의 갯수가 1000이다.
# (4, 1000)의 4는 테스트 이미지 개수를 의미한다.
print('shape of H : ', H.shape)

# 예측 값은 1000개의 class에 대한 확률 정보를 저장하고 있다.
print('예측 값 표시')
print(H)

from tensorflow.python.keras.applications.vgg16 import decode_predictions
# decode_predcitions 함수는 사람이 알아 보기 좋도록 변환해준다.
# top 매개변수를 사용하여 일부 몇 개만 확인 가능하다.
# 결과는 (class, description, probability) 순으로 출력이 된다.
result = decode_predictions(H, top=10)

print('decode_predictions 함수 실행 결과')
#print(result)

totallist=[]    # csv 파일에 저장할 데이터 리스트

# 각 이미지의 예측 결과를 출력하고, csv 파일을 위한 사전 작업을 진행한다.
for idx in range(len(mylist)):
    print('이미지 이름 : ', mylist[idx])
    #print(result[idx])

    for bbb in range(len(result[idx])):
        #print(bbb)
        sublist=[mylist[idx], result[idx][bbb][1], result[idx][bbb][2]]
        totallist.append(sublist)
    # inner for end
# outer for end

# print(totalist)
from pandas import DataFrame

mycolumn=['image', 'description', 'probability']
newframe = DataFrame(totallist, columns=mycolumn)

print('데이터 프레임으로 보여 주기')
filename='pretrainedResult.csv'
newframe.to_csv(filename, encoding='UTF-8')
print(filename + '파일이 저장됨')

from pandas.core.series import Series

mycolor = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#CCFFFF', '#CCFFDD', '#FF00CC']

for idx in range(len(result)):  # index때문에 len씀
    plt.figure(figsize=(10, 8))

    data = [] # 차트를 그리고자 하는 데이터
    captions = [] # 차트 그릴 때 x축에 입력되는 문자열(description)
    
    # result[idx]는 각 이미지에 대한 확률 정보를 저장하고 있는 list이다.
    for item in result[idx]:
        # item은 tuple인데, 1번째 요소가 description이고, 2번째 요소가 확률 값이다.
        #print(item)
        captions.append(item[1])
        data.append(100.0 * item[2])
    # inner for end

    chartdata = Series(data, index=captions)
    chartdata.plot(kind='bar', rot=12, color=mycolor)
    plt.title('이미지' + mylist[idx] + '분류 결과')
    plt.ylim([-10, 100])
    plt.grid(True)

    filename=img_source + mylist[idx].split('.')[0] + '.png'
    plt.savefig(filename)
    print(filename + '파일이 저장됨')
# outer for end

print('finished')