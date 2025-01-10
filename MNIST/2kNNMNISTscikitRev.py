from sklearn.neighbors import KNeighborsClassifier
#import functions as fs
import functionsRev as fs
import numpy as np
import time
import pdb

t1=time.time()
train,test=fs.init_data() # read data file:train.bin, test.bin


trainSet,testSet=fs.data_ready2(train,test) # 일부분의  data 가져오기
label=np.tile(np.arange(0,10),(300,1))
pdb.set_trace()

knn=KNeighborsClassifier(n_neighbors=10,weights="distance",metric="euclidean")
#knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(trainSet,label.T.flatten()) # T : transpose
result=knn.predict(testSet)
result=result.reshape(10,100).T


acc,pre,rec,f1=fs.calcMeasure(result) # 성능지표 계산, acc.sum()해볼것
t2=time.time()
print(acc,pre,rec,f1) # class 별 성능
print(t2-t1)
print('Acc=',acc.mean())
print('F1=',f1.mean())

