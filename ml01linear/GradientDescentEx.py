# 경사 하강법
x, y, m = 1.0, 1.0, 1 # m은 도수

# 새로운 w(가중치)를 구해 주는 함수
def calc(w, alpha):
    result = w - alpha * (1/m) * (w*x - y) * x
    return result

# w(가중치)가 1.01미만이면 강제로 종료
def iteration(w, alpha):
    cnt = 0 # 수시로 변하는 카운터 변수
    totaldata = [] # counter, data를 저장할 리스트
    counter = [] # 카운터 변수를 저장할 카운터 리스트
    data = [] # w 값의 추이를 저장할 리스트

    while(True):
        if w < 1.01 : # 실제 정답은 1.0이지만 1.01에서 마무리 하겠다.
            message= '학습률이 ' + str(alpha) + '일 때 반복 회수는' + str(cnt) + '입니다'
            print(message)
            break
        cnt += 1
        w=calc(w, alpha)
        counter.append(cnt)
        data.append(w)
        #print('cnt : ', cnt, 'w :' ,w)

    totaldata.append(counter)
    totaldata.append(data)
    return totaldata

import matplotlib.pyplot as plt
plt.rc('font', family="MALGUN GOTHIC")

alpha_list = [0.0001, 0.01, 0.1] # 학습율

def makeChart(draw_chart, index, learning_rate):
    plt.figure()
    plt.plot(draw_chart[0], draw_chart[1], color='b', linewidth=1, linestyle='solid')
    filename = 'GradientDescentEx' + str(index).zfill(2) + '.png'
    plt.title('학습률 :' +str(learning_rate) + ', 반복 회수 : ' +str(len(draw_chart[0])))
    plt.savefig(filename)

for idx in range(len(alpha_list)):
    w = 5.0
    chartdata = iteration(w, alpha_list[idx])
    #print('chardata : ', chartdata)
    makeChart(chartdata, idx+1, alpha_list[idx])
    print('-'*30)

print('finished')

