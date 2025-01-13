import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
import time

import functions as fs 


t1=time.time()
train,test=fs.init_data() # read data file:train.bin, test.bin


trainSet,testSet=fs.data_ready1(train,test) # 일부분의  data 가져오기
trainSetf1, testSetf1=fs.feat1(trainSet,testSet)

result=fs.nBayes(trainSetf1,testSetf1,5)

acc,pre,rec,f1=fs.calcMeasure(result) # 성능지표 계산
t2=time.time()
print(acc,pre,rec,f1) # class 별 성능
print(t2-t1)
print('Acc=',acc.mean())
print('F1=',f1.mean())
