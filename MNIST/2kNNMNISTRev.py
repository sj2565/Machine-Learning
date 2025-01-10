import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import time

#import functions as fs
import functionsRev as fs


t1=time.time()
train,test=fs.init_data() # read data file:train.bin, test.bin

##print(len(train))
##print(len(train[0]))
##print(train[0][0].shape)
##
##for i in range(10):
##    plt.subplot(2,5,i+1),plt.imshow(train[i][0],'gray')
##    plt.axis('off')
##    print(len(train[i]))
##
##plt.show()

trainSet,testSet=fs.data_ready2(train,test) # 일부분의  data 가져오기

result=fs.knn(trainSet,testSet,k=200)

acc,pre,rec,f1=fs.calcMeasure(result) # 성능지표 계산
t2=time.time()
print(acc,pre,rec,f1) # class 별 성능
print(t2-t1)
print('Acc=',acc.mean())
print('F1=',f1.mean())

