DATA_IN_PATH = './data_in/'

import os
print("파일 크기 : ")
for filename in os.listdir(DATA_IN_PATH):
    if 'txt' in filename :
        print(filename.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + filename) / 1000000, 2)) + 'MB')

mini_file_mode=True # 작은 파일을 실행할 것인가?

import pandas as pd
if mini_file_mode==True :
    train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train_mini.txt', header = 0, delimiter = '\t', quoting = 3)
else:
    train_data = pd.read_csv(DATA_IN_PATH + 'ratings_train.txt', header = 0, delimiter = '\t', quoting = 3)

print('학습용 데이터 일부')
print(train_data.head())

print('전체 학습 데이터의 개수: {}'.format(len(train_data)))
train_length = train_data['document'].astype(str).apply(len)
print('행별 문자열의 길이')
print(train_length.head())

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(12, 5))
plt.hist(train_length, bins=200, alpha=0.5, color= 'r', label='word')
plt.yscale('log', nonpositive='clip')
plt.title('행별 문자열 길이(log 히스토 그램)')
plt.xlabel('Length of review')
plt.ylabel('Number of review')
filename = 'korean-preprocessing-01.png'
plt.savefig(filename)
print(filename+' 파일 저장됨')

import numpy as np
print('리뷰 길이 최대 값: {}'.format(np.max(train_length)))
print('리뷰 길이 최소 값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균 값: {:.2f}'.format(np.mean(train_length)))
print('리뷰 길이 표준 편차: {:.2f}'.format(np.std(train_length)))
print('리뷰 길이 중간 값: {}'.format(np.median(train_length)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('리뷰 길이 제 1 사분위: {}'.format(np.percentile(train_length, 25)))
print('리뷰 길이 제 3 사분위: {}'.format(np.percentile(train_length, 75)))

plt.figure(figsize=(12, 5)) # showmeans: 평균값을 마크함
plt.boxplot(train_length, labels=['counts'], showmeans=True)
plt.title('박스 플롯 생성')
filename = 'korean-preprocessing-02.png'
plt.savefig(filename)
print(filename+' 파일 저장됨')

train_review = [review for review in train_data['document'] if type(review) is str]
count=5
print('앞 '+str(count)+'개의 행 보기')
print(train_review[0:count])

from wordcloud import WordCloud
plt.figure(figsize=(20, 15))
cloud = WordCloud().generate(' '.join(train_review))
plt.imshow(cloud)
plt.axis('off')
filename = 'korean-preprocessing-03.png'
plt.savefig(filename)
print(filename+' 파일 저장됨')

fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6, 3)

import seaborn as sns
print('긍정과 부정에 대한 빈도를 시각화')
plt.figure()
sns.countplot(train_data['label'])

# 그래프 그릴때 다음과 같은 경고 메시지가 나옵니다. 확인 요망
# C:\Anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
#   warnings.warn(
# korean-preprocessing-04.png 파일 저장됨

filename = 'korean-preprocessing-04.png'
plt.savefig(filename)
print(filename+' 파일 저장됨')

# 'label' 컬럼은 0또는 1의 값을 가지고 있습니다.
print("긍정 리뷰 개수: {}".format(train_data['label'].value_counts()[1]))
print("부정 리뷰 개수: {}".format(train_data['label'].value_counts()[0]))

print('행별 단어의 갯수')
train_word_counts = train_data['document'].astype(str).apply(lambda x:len(x.split(' ')))
print('앞 '+str(count)+'개의 단어 개수 파악하기')
print(train_word_counts[0:count])

plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=100, facecolor='r',label='train')
plt.title('Log-Histogram of word count in review', fontsize=15)
plt.yscale('log', nonpositive='clip')
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Number of reviews', fontsize=15)
filename = 'korean-preprocessing-05.png'
plt.savefig(filename)
print(filename+' 파일 저장됨')

print('리뷰 단어 개수 최대 값: {}'.format(np.max(train_word_counts)))
print('리뷰 단어 개수 최소 값: {}'.format(np.min(train_word_counts)))
print('리뷰 단어 개수 평균 값: {:.2f}'.format(np.mean(train_word_counts)))
print('리뷰 단어 개수 표준 편차: {:.2f}'.format(np.std(train_word_counts)))
print('리뷰 단어 개수 중간 값: {}'.format(np.median(train_word_counts)))
# 사분위의 대한 경우는 0~100 스케일로 되어있음
print('리뷰 단어 개수 제 1 사분위: {}'.format(np.percentile(train_word_counts, 25)))
print('리뷰 단어 개수 제 3 사분위: {}'.format(np.percentile(train_word_counts, 75)))

qmarks = np.mean(train_data['document'].astype(str).apply(lambda x: '?' in x))  # 물음표가 구두점으로 쓰임
fullstop = np.mean(train_data['document'].astype(str).apply(lambda x: '.' in x))  # 마침표

print('물음표가 있는 리뷰 : {:.2f}%'.format(qmarks * 100))
print('마침표가 있는 리뷰 : {:.2f}%'.format(fullstop * 100))

print('\n앞 '+str(count)+'개 정규식으로 전처리하기')
import re
for idx in range(count):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", train_data['document'][idx])
    # print(review_text)

from PyKomoran import Komoran
komoran = Komoran('STABLE')

print("train_data['document'][0]")
print(train_data['document'][0])

# get_plain_text 함수는 '형태소/품사'라는 형태로 태깅된 결과를 반환해 줍니다.
# 품사표 참조 사이트) https://komorandocs.readthedocs.io/ko/latest/firststep/postypes.html
txt2 = komoran.get_plain_text(train_data['document'][0])
# print('txt2')
# print(txt2)

# '\s'는 space를 표현하며 공백 문자를 의미한다.
txt = komoran.get_plain_text(train_data['document'][0]).split('\s')
# print('txt')
# print(txt)

# print(review_text)
review_text = komoran.morphes(review_text)
# print('불용어 처리 이전의 결과 보기')
# print(review_text)

stop_words = set(['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한', '펙'])
clean_review = [token for token in review_text if not token in stop_words]
# print('불용어 처리 이후의 결과 보기')
# print(clean_review)

# 전체 데이터를 위한 전처리 함수를 다음과 같이 정의합니다.
def preprocessing(review, konlpy, remove_stopwords=False, stop_words=[]):
    # 함수의 인자는 다음과 같다.
    # review : 전처리할 텍스트
    # konlpy : konlpy 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
    # remove_stopword : 불용어를 제거할지 선택 기본값은 False
    # stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트

    # 1. 한글 및 공백 문자를 제외한 문자 모두를 제거합니다.
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", review)

    # 2. konlpy 객체를 활용해서 형태소 단위로 나누는 작업을 수행합니다.
    if konlpy != None:
        word_review = konlpy.morphes(review_text)
        if remove_stopwords: # 불용어 제거(선택적)하기
            word_review = [token for token in word_review if not token in stop_words]
        return word_review
    else :
        return ""

clean_train_review = [] # 전처리된 훈련용 데이터 셋

print('training data 전처리 작업이 진행중입니다. 잠시만 기다려 주세요.')
for review in train_data['document']:
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        # print('리뷰 :' + review)
        clean_train_review.append(preprocessing(review, komoran, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_train_review.append([])  # string이 아니면 비어있는 값 추가

print('training data 전처리 작업이 완료 되었습니다.')
if mini_file_mode==True :
    print('clean_train_review')
    print(clean_train_review)

if mini_file_mode==True :
    test_data = pd.read_csv(DATA_IN_PATH + 'ratings_test_mini.txt', header=0, delimiter='\t', quoting=3)
else:
    test_data = pd.read_csv(DATA_IN_PATH + 'ratings_test.txt', header=0, delimiter='\t', quoting=3)

clean_test_review = [] # 전처리된 테스트용 데이터 셋

print('testing data 전처리 작업이 진행중입니다. 잠시만 기다려 주세요.')
for review in test_data['document']:
    # 비어 있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        try:
            clean_test_review.append(preprocessing(review, komoran, remove_stopwords=True, stop_words=stop_words))
        except Exception as e:
            # print(e)
            pass
        continue
    else:
        clean_test_review.append([])  # string이 아니면 비어있는 값 추가

print('testing data 전처리 작업이 완료 되었습니다.')
if mini_file_mode==True :
    print('clean_test_review')
    print(clean_test_review)

from tensorflow.python.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts(clean_train_review)

word_vocab = tokenizer.word_index  # 단어 사전 형태
if mini_file_mode==True :
    print('word_vocab')
    print(word_vocab)

train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

if mini_file_mode==True :
    print('train_sequences')
    print(train_sequences)
    print('test_sequences')
    print(test_sequences)

MAX_SEQUENCE_LENGTH = 8  # 문장 최대 길이

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  # 학습 데이터를 벡터화
y_train = np.array(train_data['label'])  # 학습 데이터의 라벨

x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  # 테스트 데이터를 벡터화
y_test = np.array(test_data['label'])  # 테스트 데이터의 라벨

if mini_file_mode==True :
    print('x_train')
    print(x_train)
    print('y_train')
    print(y_train)
    print('x_test')
    print(x_test)
    print('y_test')
    print(y_test)

DATA_IN_PATH = './data_in/'
X_TRAIN_DATA = 'nsmc_train_input.npy'
Y_TRAIN_DATA = 'nsmc_train_label.npy'
X_TEST_DATA = 'nsmc_test_input.npy'
Y_TEST_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'

import os

# 저장하는 디렉토리가 존재하지 않으면 생성
if not os.path.exists(DATA_IN_PATH):
    os.makedirs(DATA_IN_PATH)

# 전처리 된 학습 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + X_TRAIN_DATA, 'wb'), x_train)
np.save(open(DATA_IN_PATH + Y_TRAIN_DATA, 'wb'), y_train)

# 전처리 된 테스트 데이터를 넘파이 형태로 저장
np.save(open(DATA_IN_PATH + X_TEST_DATA, 'wb'), x_test)
np.save(open(DATA_IN_PATH + Y_TEST_DATA, 'wb'), y_test)

# 데이터 사전을 json 형태로 저장
# 단어 개수와 {"단어":"개수"} 형태의 dict 정보를 저장할 사전
data_configs = {} # 실제 파일로 저장될 예정

data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)  # vocab size 추가

import json
json.dump(data_configs, open(DATA_IN_PATH + DATA_CONFIGS, 'w'), ensure_ascii=False)

print('finished')