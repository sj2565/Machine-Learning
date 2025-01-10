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

def data_ready1(train,test,k=300):
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
        tmpl[:,i*28:(i+1)*28]=np.mean(imsi,axis=0)
        print(np.mean(imsi,axis=0).shape)
    return tmpl

def tmplMatch(tmpl, testSet):
    result = np.zeros((100,10))

    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsiTest = np.tile(testSet[i][j], (1,10))
            error = np.abs(tmpl-imsiTest)
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(), error[:,56:84].sum(), error[:,84:112].sum(), error[:,112:140].sum(),
 error[:,140:168].sum(), error[:,168:196].sum(), error[:,196:224].sum(), error[:,224:252].sum(), error[:,252:280].sum()]
            result[j,i] = np.argmin(errorSum)
    return result                  
                    
def calcMeasure(result):

    # acc = (tp+tn)/ (tp+fn+fp+tn)
    # pre = tp/ (tp+fp)
    # rec = tp/ (tp+fn)
    # f1 = 2*pre*rec/(pre+rec)
    s1, s2 = result.shape
    label = np.tile(np.arange(0,s2), (s1,1))
    pdb.set_trace()

    TP = []; TN = []; FN = []; FP = []
    for i in range(10):
##        TP.append(((result == label) & (label == i)).sum())
##        TN.append(((result == label) & (label != i)).sum())
##        FP.append(((result != label) & (result == i)).sum())
##        FN.append(((result != label) & (label == i)).sum())
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
        FP.append(((result != label) & (label == i)).sum())
        FN.append(((result == i) & (label != i)).sum())


    TP = np.array(TP); TN = np.array(TN); FN = np.array(FN); FP = np.array(FP)
    acc = (TP+TN)/(TP+TN+FP+FN)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2*pre*rec/(pre+rec)

    return acc, pre, rec, f1

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

result=tmplMatch(tmpl,testSet) # matching 결과 가져오기
print(result.shape)

acc,pre,rec,f1=calcMeasure(result) # 성능지표 계산
print(acc,pre,rec,f1) # class 별 성능
#print(acc.mean(),pre.mean(),rec.mean(),f1.mean())


