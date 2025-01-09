from PyKomoran import *
komoran = Komoran('STABLE')
print(komoran.get_plain_text("KOMORAN은 한국어 형태소 분석기입니다."))

print(komoran.morphes("KOMORAN은 한국어 형태소 분석기입니다."))

# 분석할 문장 준비
str_to_analyze = "① 파격 대한민국은 민주공화국이다. ② 대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다."

# get_nouns(): 입력 문장에서 명사만 추출합니다.
only_nouns=komoran.get_nouns(str_to_analyze)
print(only_nouns)
# # 실행 결과
# ['대한민국', '민주공화국', '대한민국', '주권', '국민', '권력', '국민']

# get_morphes_by_tags(): 입력 문장에서 주어진 품사들만 추출합니다.
print(komoran.get_morphes_by_tags(str_to_analyze, tag_list=['NNP', 'NNG', 'SF']))
# # 실행 결과
# ['대한민국', '민주공화국', '.', '대한민국', '주권', '국민', '권력', '국민', '.']

# get_plain_text(): 입력 문장 내에 형태소/품사 형태로 태그를 합니다.
print(komoran.get_plain_text(str_to_analyze))
# # 실행 결과
# ①/SW 대한민국/NNP 은/JX 민주공화국/NNP 이/VCP 다/EF ./SF ②/SW 대한민국/NNP 의/JKG 주권/NNP 은/JX 국민/NNG 에게/JKB 있/VV 고/EC ,/SP 모든/MM 권력/NNG 은/JX 국민/NNG 으로부터/JKB 나오/VV ㄴ다/EF ./SF

# get_token_list(): 입력 문장에 대해서 형태소/품사/시작지점/종료지점을 갖는 Token 자료형들을 반환받습니다.
print(komoran.get_token_list(str_to_analyze))
# # 실행 결과
# [①/SW(0,1), 대한민국/NNP(2,6), 은/JX(6,7), 민주공화국/NNP(8,13), 이/VCP(13,14), 다/EF(14,15), ./SF(15,16), ②/SW(17,18), 대한민국/NNP(19,23), 의/JKG(23,24), 주권/NNP(25,27), 은/JX(27,28), 국민/NNG(29,31), 에게/JKB(31,33), 있/VV(34,35), 고/EC(35,36), ,/SP(36,37), 모든/MM(38,40), 권력/NNG(41,43), 은/JX(43,44), 국민/NNG(45,47), 으로부터/JKB(47,51), 나오/VV(52,54), ㄴ다/EF(53,55), ./SF(55,56)]

# get_token_list(flatten=False): 입력 문장에 대해서 Token 자료형들을 반환받습니다. 이 때, 어절 단위로 나누어 반환받습니다.
print(komoran.get_token_list(str_to_analyze, flatten=False))
# # 실행 결과
# [[①/SW(0,1)], [대한민국/NNP(2,6), 은/JX(6,7)], [민주공화국/NNP(8,13), 이/VCP(13,14), 다/EF(14,15), ./SF(15,16)], [②/SW(17,18)], [대한민국/NNP(19,23), 의/JKG(23,24)], [주권/NNP(25,27), 은/JX(27,28)], [국민/NNG(29,31), 에게/JKB(31,33)], [있/VV(34,35), 고/EC(35,36), ,/SP(36,37)], [모든/MM(38,40)], [권력/NNG(41,43), 은/JX(43,44)], [국민/NNG(45,47), 으로부터/JKB(47,51)], [나오/VV(52,54), ㄴ다/EF(53,55), ./SF(55,56)]]

# get_token_list(flatten=False): 입력 문장에 대해서 Token 자료형들을 반환받습니다. 이 때, 품사 기호 대신 이름을 사용합니다.
print(komoran.get_token_list(str_to_analyze, use_pos_name=True))
# # 실행 결과


# get_list(): 입력 문장에 대해서 형태소/품사를 갖는 Pair 자료형들을 반환받습니다.
result=komoran.get_list(str_to_analyze)
print(type(result))
print(result[0:2])
print(result)
# # 실행 결과
# [①/SW, 대한민국/NNP, 은/JX, 민주공화국/NNP, 이/VCP, 다/EF, ./SF, ②/SW, 대한민국/NNP, 의/JKG, 주권/NNP, 은/JX, 국민/NNG, 에게/JKB, 있/VV, 고/EC, ,/SP, 모든/MM, 권력/NNG, 은/JX, 국민/NNG, 으로부터/JKB, 나오/VV, ㄴ다/EF, ./SF]