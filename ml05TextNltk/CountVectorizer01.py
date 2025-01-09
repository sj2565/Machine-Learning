# 차후에는 파일(csv, txt 형식)을 이용하여 처리해 보도록 할 것
content = ['우리 아버지 이름은 홍길동', '홍길동 여자 친구 이름은 심순애', '여자 친구 있나요']
            # document 1                 document 2                  document 3

# 각각의 단어의 개수를 세어서 BOW로 인코딩한 벡터(Vector)를 생성해 준다.
from sklearn.feature_extraction.text import CountVectorizer

# df : document frequency
# min_df=2 : df >= 2인 단어들만 출력. (위의 document에서 이름은, 홍길동, 여자  => 두번 이상 사용)
vectorizer = CountVectorizer(min_df=2, stop_words=['친구']) # 친구는 두번 이상 쓰였어도 출력 x
#vectorizer = CountVectorizer(min_df=1)
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
    myarray = vectorizer.transform([data]).toarray()
    print(data)
    print(myarray)  
    # 색인별로 정렬이 된 토큰들(여자, 이름은, 홍길동)을 문장에 얼마나 사용되었는지 빈도 수를 알려줌
    # ex) 우리 아버지 이름은 홍길동  => 여자 : 0, 이름은 : 1, 홍길동 : 1 쓰임
    print('-'*30)