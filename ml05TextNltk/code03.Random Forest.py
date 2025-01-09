DATA_IN_PATH = './data_in/'
TRAIN_CLEAN_DATA = 'train_clean.csv'

import pandas as pd
train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)

reviews = list(train_data['review'])

import numpy as np
y = np.array(train_data['sentiment'])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', max_features=5000)

x = vectorizer.fit_transform(reviews)

from sklearn.model_selection import train_test_split
TEST_SIZE = 0.2
RANDOM_SEED = 42
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

forest.fit(x_train, y_train)

# 검증 데이터로 정확도 측정
print('정확도 : %.4f' %(forest.score(x_test, y_test)))

# 테스트용 데이터 셋으로 해당 모델에 대한 성능을 평가한다.
TEST_CLEAN_DATA = 'test_clean.csv'
DATA_OUT_PATH = './data_out/'

test_data = pd.read_csv(DATA_IN_PATH + TEST_CLEAN_DATA)
test_reviews = list(test_data['review'])
ids=list(test_data['id'])

test_data_reviews = vectorizer.transform(test_reviews)

# 테스트 결과를 csv 파일로 저장한다.
import os
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

result = forest.predict(test_data_reviews)

output = pd.DataFrame({'id' :ids, 'sentiment' :result})
output.to_csv(DATA_OUT_PATH + 'bag_of_words_model.csv', index=False, quoting=3)
print('finished')