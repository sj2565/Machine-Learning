# 차후에는 파일(csv, txt 형식)을 이용하여 처리해 보도록 할 것
content = ['우리 아버지 이름은 홍길동', '홍길동 여자 친구 이름은 심순애', '여자 친구 있나요']
            # document 1                 document 2                  document 3

# 각각의 단어의 개수를 세어서 BOW로 인코딩한 벡터(Vector)를 생성해 준다.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=1, stop_words=['친구'])
print(type(vectorizer))

# fit 함수는 content에 대한 학습을 수행하여 단어들에 대한 사전을 만들어 준다.
matrix = vectorizer.fit(content)
print(type(matrix))

# vocabulary_ 속성은 단어 사전을 보여 준다.
# 가나다 순으로 정렬하여(출력은 가나다 순으로 안 나옴), 단어별 색인 번호를 매긴다.
print('matrix.vocabulary_')
print(matrix.vocabulary_) # 어휘 출력

print('단어 사전들을 정렬하기')
print(sorted(matrix.vocabulary_.items()))  # 가나다 순으로 중괄호로 묶어서 출력됨

print('단어(token) 목록 보기')
feature = vectorizer.get_feature_names()
print(type(feature))
print(feature)

print('불용어(stop_words)단어 목록 보기')
print(vectorizer.get_stop_words())

for data in content:
    # 색인별로 정렬이 된 토큰들에 대한 빈도 수를 출력해 준다.
    myarray = vectorizer.transform([data]).toarray()
    print(data)
    print(myarray)
    print('-'*30)

# 여자 친구 있나요
# 0(심순애) 1(아버지)  2(여자)    3(우리) 4(이름은) 5(있나요)  6(홍길동)
# [[0.     0.     0.60534851     0.    0.    0.79596054    0. ]]

# 문자열 '여자 친구 있나요' 에서 '여자' 라는 단어 보다 '있나요' 라는 단어가 희귀성이 있어서
# 더 높은 가중치를 부여 받는다.