import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

def init_data():
    with open('train.bin','rb') as f1:
        train=pickle.load(f1)

    with open('test.bin','rb') as f2:
        test=pickle.load(f2)

    return train,test

def data_ready1(train,test,k=100):
    trainSet=[]
    testSet=[]
#    pdb.set_trace() #중단점 표시
    for i in range(10):
        trainSet.append(train[i][0:k])
        testSet.append(test[i][0:100])
    return trainSet,testSet

def createTmpl(trainSet):
    tmpl=np.zeros((28,28*10))
    print(tmpl.shape)
    for i in range(10):
#        pdb.set_trace()
        imsi=np.array(trainSet[i]) # list -> ndarray 변환
#        pdb.set_trace()
        tmpl[:,i*28:(i+1)*28]=np.mean(imsi,axis=0)
        print(np.mean(imsi,axis=0).shape)
    return tmpl


train,test=init_data() # read data file:train.bin, test.bin

print(len(train))
print(len(train[0]))
print(train[0][0].shape)

for i in range(10):
    plt.subplot(2,5,i+1),plt.imshow(train[i][0],'gray')
    plt.axis('off')
    print(len(train[i]))

plt.show()

trainSet,testSet=data_ready1(train,test,300) # 일부분의  data 가져오기
print(len(trainSet[0]))
print(len(testSet[0]))

tmpl=createTmpl(trainSet) # template 가져오기
plt.imshow(tmpl)
plt.show()


