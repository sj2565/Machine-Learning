DATA_IN_PATH = './data_in/'
TRAIN_CLEAN_DATA = 'train_clean.csv'

import pandas as pd
train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
print(train_data.info())
print(train_data.columns)

reviews = list(train_data['review'])
sentiments = list(train_data['sentiment'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.0, analyzer='char', sublinear_tf=True,
                             ngram_range=(1, 3), max_features=5000)

x = vectorizer.fit_transform(reviews)
import numpy as np
y = np.array(sentiments)

feautres = vectorizer.get_feature_names()
print('토큰의 개수 : ', len(feautres))
print('토큰화 단어 목록 : ', feautres)

RANDOM_SEED = 42
TEST_SPLIT = 0.2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)

from sklearn.linear_model import LogisticRegression
lgs = LogisticRegression(class_weight='balanced')

lgs.fit(x_train, y_train)

predicted = lgs.predict(x_test)

print('정확도 : %.4f' % lgs.score(x_test, y_test))

TEST_CLEAN_DATA = 'test_clean.csv'
test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)

testDataVecs = vectorizer.transform(test_data['review'])
test_predicted = lgs.predict(testDataVecs)
print('test_predicted : ', test_predicted)

# 테스트 된 결과물 csv파일로 저장한다.
DATA_OUT_PATH = './data_out/'

import os
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

answer_dataset = pd.DataFrame({'id':test_data['id'], 'sentiment':test_predicted})
answer_dataset.to_csv(DATA_OUT_PATH + 'lgs_tfidf_answer.csv', index=False,
                      quoting=3)
print('finished')