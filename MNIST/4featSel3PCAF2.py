import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import functions as fs
import pdb

#t1=time.time()
#pdb.set_trace()
train, test=fs.init_data()
trainSet, testSet=fs.data_ready1(train,test)

dx=3
trainSetf1, testSetf1=fs.feat2(trainSet, testSet, dx)
##t1=time.time()
##result=fs.knn(trainSetf1, testSetf1, k=10)
##t2=time.time()
##print(t2-t1)
##acc, pre, rec, f1=fs.calcMeasure(result)
##print('Acc=',acc.mean())
##print('F1=',f1.mean())

trainSetf2, testSetf2=fs.pca(trainSetf1, testSetf1,45)
t1=time.time()
result=fs.knn(trainSetf2, testSetf2, k=500)
t2=time.time()
print(t2-t1)
acc, pre, rec, f1=fs.calcMeasure(result)
print('PCA_Acc=',acc.mean())
print('PCA_F1=',f1.mean())
