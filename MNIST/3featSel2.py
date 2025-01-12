import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import functions as fs
import pdb

t1=time.time()
#pdb.set_trace()
train, test=fs.init_data()
trainSet, testSet=fs.data_ready1(train,test)

dX=20; k=500
trainSetf1, testSetf1=fs.feat2(trainSet, testSet, dX)
result=fs.knn(trainSetf1, testSetf1, k)
acc, pre, rec, f1=fs.calcMeasure(result)
t2=time.time()
print(t2-t1)
print('k=',k,' dX=',dX)
print('Acc=',acc.mean())
print('F1=',f1.mean())
