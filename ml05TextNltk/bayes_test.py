import sys, math
from PyKomoran import *

class BayesianFilter :
    def __init__(self):
        # 워드 집합 : unique()한 모든 단어(word)들을 저장하고 있는 집합
        self.words = set()

        # 워드 사전 : 카테고리 당 word 빈도 수를 저장하고 있는 중첩 사전
        self.word_dict = dict()

        # 카테고리 사전 : 카테고리 빈도 수를 저장하는 사전
        self.category_dict = {}

    # 넘겨진 text를 형태소 분석하고, 해당 결과를 list 형태로 반환해 준다.
    def mysplit(self, text):
        komoran = Komoran('STABLE')
        result = komoran.get_nouns(text)
        return result

    def fit(self, text, category):
        # text는 이메일 제목, category는 카테고리('광고 메일', '일반 메일' 중에서 하나)
        word_list=self.mysplit(text)
        print('이메일 제목 : ' ,text)
        print('단어 리스트 : ' ,word_list)

        for word in word_list:
            self.inc_word(word, category)
        self.inc_category(category)

    def inc_word(self, word, category):
        # '워드 사전'에 해당 카테고리가 존재하지 않으면 추가한다.
        if not category in self.word_dict:
            self.word_dict[category]={}

        # '카테고리 사전'에 해당 카테고리가 존재하지 않으면 추가한다.
        if not word in self.word_dict[category]:
            self.word_dict[category][word] = 0
        self.word_dict[category][word] += 1

        # '워드 집합'에 대한 워드를 추가한다.
        self.words.add(word)

    def inc_category(self, category):
        # '카테고리 사전'에 해당 카테고리가 존재하지 않으면 추가한다.
        if not category in self.category_dict:
            self.category_dict[category] = 0
        self.category_dict[category] += 1

    def score(self, words, category):
        # 특정 카테고리(category)에서 해당 이메일의 형태소 목록(words)에 대한 점수를 계산한다.
        # 점수 = 카테고리의 점수 + Σ워드별_점수
        # 비율에 log 함수를 씌우면 음수 값이 나온다.
        # 이 값과 max_score와 비교하여 큰 수를 찾는다.
        
        # 카테고리_점수 = 해당카테고리값/전체카테고리총합
        bunmo = sum(self.category_dict.values())
        bunja = self.category_dict[category]
        score = math.log(bunja/bunmo)

        for word in words:
            # 해당 '워드'가 워드 사전에 존재하는 경우
            if word in self.word_dict[category]:
                bunja = self.word_dict[category][word]
            else :
                bunja = 0

            # '워드 사전'에 존재하지 않던 신규 워드는 0이 된다.
            # 확률의 곱셈 법칙 때문에 0을 막을려고 +1을 해 준다.
            bunja += 1

            bunmo = sum(self.word_dict[category].values()) + len(self.words)
            score += math.log(bunja/bunmo)
        return score

    def predict(self, checkmail):
        # checkmail은 검증을 하고자 하는 이메일 제목
        best_category = None # 어떠한 메일인지를 알려주는 문자열
        
        # 가정) 현재 시스템이 가지고 있는 최소값을 최대값이라고 가정한다
        max_score = -sys.maxsize    # 최대 점수
        words = self.mysplit(checkmail)
        score_list = [] # 카테고리별 점수를 저장할 리스트

        for category in self.category_dict.keys():
            score = self.score(words, category)
            score_list.append((category, score))

            print('현 카테고리 : %s' % (category))
            print('점수 : %.f' % (score))
            print('최고 점수 : %f' % (max_score))

            print(category, score, max_score)

            if score > max_score: # 현재 카테고리의 점수가 최대 점수보다 크면
                max_score = score # 현재 점수를 최대 점수로 변경
                best_category = category # 현재 카테고리 정보를 메일 유형 변수에 저장

        # 메일의 유형과 카테고리별 점수 목록을 tuple 형태로 반환해 줌
        return best_category, score_list
    
    def showInfo(self):
        print('self.words')
        print(self.words)

        print('self.word_dict')
        print(self.word_dict)

        print('self.category_dict')
        print(self.category_dict)
# end class BayesianFilter

bf = BayesianFilter()   # 객체 생성

# 다음 데이터를 이용하여 텍스트 학습을 진행한다.
bf.fit("파격 세일", "광고 메일")
bf.fit("오늘 일정 확인", "일반 메일")
bf.fit("파격 세일", "광고 메일")
bf.fit("쿠폰 선물 & 무료 배송", "광고 메일")
bf.fit("신세계 백화점 세일", "광고 메일")
bf.fit("봄과 함께 찾아온 따뜻한 신제품 소식", "광고 메일")
bf.fit("인기 제품 기간 한정 세일", "광고 메일")

bf.fit("프로젝트 진행 상황 보고", "일반 메일")
bf.fit("계약 잘 부탁드립니다", "일반 메일")
bf.fit("회의 일정이 등록되었습니다", "일반 메일")
bf.fit("오늘 일정이 없습니다", "일반 메일")

bf.showInfo()

# 다음 데이터에 대하여 예측을 해 본다.
email='재고 정리 할인 무료 배송'
pre, scorelist = bf.predict(email)
print("메일 제목이 '%s'인 메일은 '%s'입니다. " %(email, pre))
print(scorelist)

print('finished')
