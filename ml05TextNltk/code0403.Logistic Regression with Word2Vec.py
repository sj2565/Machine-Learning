# Logistic Regression Example with Word2Vec
'''
word2vec은 단어로 표현된 리스트 형식의 데이터가 필요합니다.
따라서, 정제된 텍스트 파일인 'train_clean.csv'을 실습에 사용하도록 합니다.

'''
# import os
# import re
# from bs4 import BeautifulSoup
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize


DATA_IN_PATH='./data_in/'
DATA_OUT_PATH='./data_out/'
TRAIN_CLEAN_DATA='train_clean.csv' # 정제된 텍스트 파일

RANDOM_SEED=42
TEST_SPLIT=0.2

import pandas as pd
train_data=pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)

reviews=list(train_data['review'])
sentiments=list(train_data['sentiment'])

# word2vec 모델은 단어를 이용하여 처리해 주므로 'review'를 'word'로 변환해야 합니다.
sentences=[] # 단어 목록이 들어갈 리스트 변수(중첩 리스트)
for review in reviews:
    # review.split()는 한 행당 단어 목록을 리스트로 만들어 줍니다.
    sentences.append(review.split())

print('\n# 앞 요소 2개만 출력해보기')
# [['stuff', 'going', 'moment', 'mj', ...], ['classic', 'war', 'worlds', 'timothy', ...]]
print(sentences[0:2])

print('\n#word2vec 모델 학습시 필요한 로깅 지정하기')
print('\n#출력 형식은 시간 : 레벨 : 로깅 메시지~입니다.')
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('\n#word2vec 모델을 위한 하이퍼 파라미터 설정하기')
num_features=300
min_word_count=40 # 무시할 총 사용 빈도의 최대 값
num_workers=4
context=10 # 윈도우의 크기
downsampling=1e-3

# pip install gensim
from gensim.models import word2vec
model=word2vec.Word2Vec(sentences, workers=num_workers,
    min_count=min_word_count,
    window=context, sample=downsampling)

# # model=word2vec.Word2Vec(sentences, workers=num_workers,
# #     size=num_features, min_count=min_word_count,
# #     window=context, sample=downsampling)
#

import numpy as np

# AttributeError: The index2word attribute has been replaced by index_to_key since Gensim 4.0.0.
def get_features(words, model, num_features):
    feature_vector=np.zeros((num_features), dtype=np.float32)

    num_words=0
    index2word_set=set(model.wv.index_to_key)

    for w in words:
        if w in index2word_set:
            num_words += 1
            feature_vector=np.add(feature_vector, model[w])

    feature_vector=np.divide(feature_vector, num_words)
    return feature_vector


def get_dataset(reviews, model, num_features):
    dataset=list()

    for s in reviews:
        dataset.append(get_features(s, model, num_features))

    reviewFeatureVecs=np.stack(dataset)

    return reviewFeatureVecs

print('\n# 앞 요소 2개만 출력해보기')
test_data_vecs=get_dataset(sentences, model, num_features)
print(test_data_vecs)

from sklearn.model_selection import train_test_split
import numpy as np

X=test_data_vecs
y=np.array(sentiments)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

from sklearn.linear_model import LogisticRegression

lgs=LogisticRegression(class_weight='balanced')
lgs.fit(X_train, y_train)

predicted=lgs.predict(X_test)
from sklearn import metrics

fpr, tpr, _=metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc=metrics.auc(fpr, tpr)

print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  # checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)

TEST_CLEAN_DATA='test_clean.csv'

test_data=pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)

test_review=list(test_data['review'])

print('test_data.head(5) :', test_data.head(5))

test_sentences=list()
for review in test_review:
    test_sentences.append(review.split())

test_data_vecs=get_dataset(test_sentences, model, num_features)

test_predicted=lgs.predict(test_data_vecs)

ids=list(test_data['id'])

answer_dataset=pd.DataFrame({'id': ids, 'sentiment': test_predicted})

import os
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_w2v_answer.csv', index=False, quoting=3)

print('finished')