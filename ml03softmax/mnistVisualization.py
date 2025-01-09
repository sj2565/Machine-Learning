import matplotlib.pyplot as plt
chartdata = [0.9249, 0.9226, 0.9798, 0.9733, 0.9750]
labels = ['case01', 'case02', 'case03', 'case04', 'case05']

import numpy as np
x = np.arange(len(chartdata))

plt.rc('font', family='MALGUN GOTHIC')
plt.bar(x, chartdata, color='y')
plt.xticks(x, labels)
plt.title('케이스별 정확도 그래프')

filename = 'mnistVisualization.png'
plt.savefig(filename)
print(filename + '파일 저장됨')