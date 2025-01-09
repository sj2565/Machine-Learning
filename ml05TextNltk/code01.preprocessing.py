import zipfile

import numpy as np

DATA_IN_PATH = './data_in/'

# zip 파일을 압축 해제 한다.
# file_list = ['labeledTrainData.tsv.zip', 'testData.tsv.zip', 'unlabeledTrainData.tsv.zip']
#
# for file in file_list:
#     onefile = zipfile.ZipFile(DATA_IN_PATH + file, mode='r')
#     onefile.extractall(DATA_IN_PATH)
#     onefile.close()
# # end for

import os
#
# for file in os.listdir(DATA_IN_PATH):
#     if 'tsv' in file and 'zip' not in file :
#         filesize = round(os.path.getsize(DATA_IN_PATH + file)/1000000, 2)
#         # ljust(30)은 30자리를 확보하고, 왼쪽 기준으로 정렬해라
#         message = file.ljust(30) + str(filesize) + 'MB'
#         print(message)
#
import pandas as pd
# train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
# print('train_data.head()')
# print(train_data.head())
#
# print('훈련 데이터 개수 : {}'.format(len(train_data)))
#
# print('review 컬럼의 문자 길이를 확인해 본다')
# train_length=train_data['review'].apply(len)
# print('train_data.head()')
# print(train_data.head())
#
# import matplotlib.pyplot as plt
# plt.rc('font', family='MALGUN GOTHIC')
#
# print('히스토그램 그리기')
# plt.figure(figsize=(12, 5))
# plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
# #plt.yscale('log', nonpositive='clip')
# plt.title('log-histogram of length of reviews')
# plt.xlabel('length of review')
# plt.ylabel('nubmber of review')
# filename = 'code01.preprocessing_01.png'
# plt.savefig(filename)
# print(filename + '파일이 저장됨')
#
# print('reveiw 컬럼 문자 길이에 대한 간략한 통게 정보를 확인해 본다.')
# import numpy as np
# print('리뷰 길이 최대 값 : {}'.format(np.max(train_length)))
# print('리뷰 길이 최소 값 : {}'.format(np.min(train_length)))
# print('리뷰 길이 평균 값 : {:.2f}'.format(np.mean(train_length)))
# print('리뷰 길이 표준 편차 : {:.2f}'.format(np.std(train_length)))
# print('리뷰 길이 중간 값 : {}'.format(np.median(train_length)))
#
# print('사분위수 정보를 확인해 본다.')
# print('리뷰 길이 제1사분위 값 : {}'.format(np.percentile(train_length, 25)))
# print('리뷰 길이 제3사분위 값 : {}'.format(np.percentile(train_length, 75)))
#
# print('상자 수염(Boxplot) 그래프 그리기')
# plt.figure(figsize=(12, 5))
# # showmeans = True는 평균 값을 표시하겠다.
# plt.boxplot(train_length, labels=['counts'], showmeans=True)
# plt.title('box plot of review length')
# filename = 'code01.preprocessing_02.png'
# plt.savefig(filename)
# print(filename + '파일이 저장됨')
#
# #from wordcloud import WordCloud    워드 클라우드 모듈 없음
# # clouddata = " ".join(train_data['review'])
# # cloud = WordCloud(width=800, height=600).generate(clouddata)
# # plt.figure(figsize=(20, 15))
# # plt.imshow(cloud)
# # plt.axis('off')
# # filename='code01.preprocessing_03.png'
# # plt.savefig(filename)
# # print(filename + '파일이 저장됨')
#
# print('긍정과 부정의 데이터 분포를 확인한다.')
# # value_counts() 메소드는 각 항목에 대한 빈도를 구한 다음, 빈도가 큰 것 부터 보여주는 series.
# bindo=train_data['sentiment'].value_counts()
# print(bindo)
# print('postive 리뷰 개수 : {}'.format(bindo[1]))
# print('negative 리뷰 개수 : {}'.format(bindo[0]))
#
# import seaborn as sns
# fig, axe=plt.subplots(ncols=1)
# # countplot 함수는 항목별 빈도를 토대로 막대 그래프를 그려 준다.
# sns.countplot(train_data['sentiment'])
# plt.title('긍정, 부정 데이터의 분포')
# filename='code01.preprocessing_04.png'
# plt.savefig(filename)
# print(filename + '파일이 저장됨')
#
# print('리뷰의 단어들에 대한 정보를 확인한다.')
# split_lambda = lambda x : len(x.split(' ')) # 공백으로 나눈 후 단어 개수 구하는 함수
# train_word_counts = train_data['review'].apply(split_lambda)
#
# print('리뷰의 단어 개수의 일부 데이터 확인하기')
# print('train_word_counts.head()')
# print(train_word_counts.head())
#
# print('리뷰 단어 개수 히스토그램 그리기')
# plt.figure(figsize=(15, 6))
# plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
# plt.yscale('log', nonpositive='clip')
# plt.title('각 리뷰의 단어 개수 분포', fontsize=12)
# plt.xlabel('length of word', fontsize=12)
# plt.ylabel('nubmber of review', fontsize=12)
# filename = 'code01.preprocessing_05.png'
# plt.savefig(filename)
# print(filename + '파일이 저장됨')
#
# print('리뷰 단어 개수에 대한 간략한 통게 정보를 확인해 본다.')
# print('단어 개수 최대 값 : {}'.format(np.max(train_word_counts)))
# print('단어 개수 최소 값 : {}'.format(np.min(train_word_counts)))
# print('단어 개수 평균 값 : {:.2f}'.format(np.mean(train_word_counts)))
# print('단어 개수 표준 편차 : {:.2f}'.format(np.std(train_word_counts)))
# print('단어 개수 중간 값 : {}'.format(np.median(train_word_counts)))
#
# print('사분위수 정보를 확인해 본다.')
# print('단어 개수 제1사분위 값 : {}'.format(np.percentile(train_word_counts, 25)))
# print('단어 개수 제3사분위 값 : {}'.format(np.percentile(train_word_counts, 75)))
#
# # 특수 문자, 대소문자, 마침표, 물음표, 숫자 등에 대한 비율
# review=train_data['review']
# question=np.mean(review.apply(lambda x: '?' in x)) # 물음표
# fullstop=np.mean(review.apply(lambda x: '?' in x)) # 마침표
# capital_first=np.mean(review.apply(lambda x: x[0].isupper() in x)) # 첫 글자가 대문자인가?
# capitals=np.mean(review.apply(lambda x: max([y.isupper() for y in x]))) # 대문자가 존재하는가?
# digit=np.mean(review.apply(lambda x: max([y.isdigit() for y in x]))) # 숫자가 존재하는가?
#
# print('물음표가 있는 review : {:.2f}%'.format(question))
# print('마침표가 있는 review : {:.2f}%'.format(fullstop))
# print('첫 글자가 대문자인 review : {:.2f}%'.format(capital_first))
# print('대문자 review : {:.2f}%'.format(capitals))
# print('숫자가 있는 review : {:.2f}%'.format(digit))

print('데이터 전처리를 수행합니다')
train_data = pd.read_csv(DATA_IN_PATH + 'labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

# 1번째 학습용 데이터
review=train_data['review'][0]
print(review)

from bs4 import BeautifulSoup
# pip install html5lib 설치
# html5lib를 사용하면 html 태그를 자동으로 제거해 준다.
review_text = BeautifulSoup(review, 'html5lib').get_text()

import re  
# sub() : subsitute(치환)
# [^a-zA-Z] : 알파벳을 제외한 나머지는 모두 공백으로 처리하겠습니다
regEx = '[^a-zA-Z]'
review_text=re.sub(regEx, ' ', review_text)

print('정규 표현식 및 html 태그 제거 후의 결과 보기')
print('콤마, 마침표, 특수 문자, 숫자 등이 보이지 않아야 된다')
review_text=review_text.lower() # 모든 단어 소문자화
print(review_text)

print('불용어를 제거하도록 한다')
# nltk: National Language Toolkit (자연어 처리 툴킷)
from nltk.corpus import stopwords
# stop_words는 영어를 위한 불용어 집합(set)이다.
stop_words=set(stopwords.words('english'))
print('stop_words')
print(stop_words)

words=review_text.split()   # 반환 타입은 list이다.
words=[w for w in words if not w in stop_words]
print('불용어 제거 후 결과물')
print(words)

clean_review=' '.join(words)
print('전처리된 단어들을 하나의 텍스트로 다시 합친다')
print(clean_review)

# 전처리를 수행해주는 범용 사용자 정의 함수를 작성한다.
def preprocessing(review, remove_stopwords = False):
    # review : 분석할 1줄의 긴 문장
    # remove_stopwords : 불용어 제거를 할 것인지의 여부
    # html 태그 처리
    reveiw_text = BeautifulSoup(review, "html5lib").get_text()
    # 정규 표현식을 이용한 처리
    review_text = re.sub(regEx, ' ', reveiw_text)
    # 소문자화 후에 공백으로 텍스트 분리시켜 list를 생성한다
    words = review_text.lower().split() # 메소드 체이닝

    if remove_stopwords: # 불용어를 처리하는 경우
        stop_words = set(stopwords.words('english'))    # 불용어 목록
        words = [w for w in words if not w in stop_words]   # 불용어는 제외시키기
    # end if

    clean_review = ' '.join(words)

    return clean_review
# end def preprocessing

print('모든 행에 대하여 전처리 중이다. 잠시만 기달')
# 전처리가 완료된 훈련용 데이터 셋
clean_train_reviews = []
for review in train_data['review']:
    clean_train_reviews.append(preprocessing(review, True))
print('모든 행에 대하여 전처리가 완료')

print('# 전처리된 0번째 데이터 보기')
print(clean_train_reviews[0])

print('# 전처리된 데이터를 sentiment 컬럼과 함께 csv파일로 저장한다.')
mydict={'review': clean_train_reviews, 'sentiment': train_data['sentiment']}
clean_train_df = pd.DataFrame(mydict)
print(clean_train_df.head())

filename=DATA_IN_PATH + 'train_clean.csv'
clean_train_df.to_csv(filename, index=False)
print(filename + '파일이 저장됨')

from tensorflow.python.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()

# fit_on_texts 함수는 각 단어들에 대항 색인을 만들어 주는 함수이다.
tokenizer.fit_on_texts(clean_train_reviews)

# texts_to_sequences 함수는 각각의 문자열에 대하여 정수 인덱스 목록을 list 형태로 반환해 준다.
text_sequences=tokenizer.texts_to_sequences(clean_train_reviews)
print('text_sequences[0]')
print(text_sequences[0])

# word_index 속성은 단어들에 대하여 단어 사전을 만들어 준다.
word_vocab = tokenizer.word_index

print('전체 단어의 개수 : ', len(word_vocab) + 1)

print('너무 많으므로 앞 10개만 출력해 본다.')
idx=0
for key, value in word_vocab.items():
    idx+=1
    print(key + ':' +str(value))
    if idx == 10:
        break
# end for

# 단어 사전과 전체 단어 개수를 json 파일 형식으로 저장한다.
data_config={}
data_config['vocab']=word_vocab # 단어 사전
data_config['vocab_size']=len(word_vocab) # 전체 단어 개수

import json
json.dump(data_config, open(DATA_IN_PATH + 'data_configs.json', 'w'), ensure_ascii=False)

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 174   # 단어 길이의 평균 값을 취함
train_inputs=pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('train_inputs.shape : ', train_inputs.shape)

import numpy as np
train_labels = np.array(train_data['sentiment'])
print('train_labels.shape : ', train_labels.shape)

# 전처리된 데이터를 넘파이 파일 형식으로 저장하기
np.save(open(DATA_IN_PATH + 'train_input.npy', 'wb'), train_inputs)
np.save(open(DATA_IN_PATH + 'train_labels.npy', 'wb'), train_labels)

# 테스트용 데이터도 동일하게 처리한다.
test_data=pd.read_csv(DATA_IN_PATH + 'testData.tsv', header=0, delimiter='\t', quoting=3)

clean_test_reviews=[]
for review in test_data['review']:
    clean_train_reviews.append(preprocessing(review, True))

mydict={'review': clean_test_reviews, 'id':test_data['id']}
clean_test_df = pd.DataFrame(mydict)

filename=DATA_IN_PATH+ 'test_clean.csv'
clean_test_df.to_csv(filename, index=False)
print(filename + '파일이 저장됨')

text_sequences=tokenizer.texts_to_sequences(clean_test_reviews)
test_inputs=pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

np.save(open(DATA_IN_PATH + 'test_inputs.npy', 'wb'), test_inputs)

test_id = np.array(test_data['id'])
np.save(open(DATA_IN_PATH + 'test_id.npy', 'wb'), test_id)
