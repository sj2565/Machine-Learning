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

def data_ready2(train,test,k=300):
    trainSetf=np.zeros((k*10,28*28))
    testSetf=np.zeros((100*10,28*28))

    for i in range(len(train)):
        for j in range(k):
            trainSetf[i*k+j,:]=train[i][j].flatten()
    for i in range(len(test)):
        for j in range(100):
            testSetf[i*100+j,:]=test[i][j].flatten()
    return trainSetf,testSetf

def knn(trainSet, testSet, k):

    trS1, trS2=trainSet.shape
    teS1, teS2=testSet.shape
    trS3=int(trS1/10)
    teS3=int(teS1/10)
    
    label=np.tile(np.arange(0,10),(teS3,1))
    result=np.zeros((teS3,10))

    for i in range(teS1):
        imsi=np.sum((trainSet-testSet[i,:])**2,axis=1)
        #pdb.set_trace()
        no=np.argsort(imsi)[0:k]
        hist,bins=np.histogram(no//trS3,np.arange(-0.5,10.5,1))
        result[i%teS3,i//teS3]=np.argmax(hist)
    return result

def createTmpl(trainSet):
    tmpl=np.zeros((28,28*10))
    print(tmpl.shape)
    for i in range(10):
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

    TP = []; TN = []; FN = []; FP = []
    for i in range(10):
##        TP.append(((result == label) & (label == i)).sum())
##        TN.append(((result == label) & (label != i)).sum())
##        FP.append(((result != label) & (result == i)).sum())
##        FN.append(((result != label) & (label == i)).sum())
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
#        FP.append(((result != label) & (label == i)).sum())
#        FN.append(((result == i) & (label != i)).sum())
# Below two lines are corrected by S.-H. Oh
        FN.append(((result != label) & (label == i)).sum())
        FP.append(((result == i) & (label != i)).sum())


    TP = np.array(TP); TN = np.array(TN); FN = np.array(FN); FP = np.array(FP)
    acc = (TP+TN)/(TP+TN+FP+FN)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2*pre*rec/(pre+rec)

    return acc, pre, rec, f1

def feat1(trainSet, testSet):

    trS1=len(trainSet); trS2=len(trainSet[0])
    teS1=len(testSet); teS2=len(testSet[0])

    trainSetf=np.zeros((trS1*trS2,5))
    testSetf=np.zeros((teS1*teS2,5))

    for i in range(trS1):
        for j in range(trS2):
            imsi=trainSet[i][j]
            imsi=np.where(imsi!=0)
            imsi2=np.mean(imsi,1)
            imsi3=np.cov(imsi)
            trainSetf[i*trS2+j,:]=np.array([imsi2[0],imsi2[1], imsi3[0,0], imsi3[0,1], imsi3[1,1]])

            
    for i in range(teS1):
        for j in range(teS2):
            imsi=testSet[i][j]
            imsi=np.where(imsi!=0)
            imsi2=np.mean(imsi,1)
            imsi3=np.cov(imsi)
            testSetf[i*teS2+j,:]=np.array([imsi2[0],imsi2[1], imsi3[0,0], imsi3[0,1], imsi3[1,1]])

    return trainSetf, testSetf

def feat2(trainSet, testSet, dX):
    size=trainSet[0][0].shape[0]; s=size-dX+1
    trS1=len(trainSet); trS2=len(trainSet[0])
    teS1=len(testSet); teS2=len(testSet[0])
    trainImsi=np.zeros((trS1*trS2,s,s)); testImsi=np.zeros((teS1*teS2,s,s))
    trainSetf=np.zeros((trS1*trS2,s*s)); testSetf=np.zeros((teS1*teS2,s*s))

    for i in range(trS1):
        for j in range(trS2):
            imsi=trainSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    trainImsi[i*trS2+j,ii,jj]=imsi[ii:dX+ii,jj:dX+jj].sum()
            trainSetf[i*trS2+j,:]=trainImsi[i*trS2+j,::].flatten()

    for i in range(teS1):
        for j in range(teS2):
            imsi=testSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    testImsi[i*teS2+j,ii,jj]=imsi[ii:dX+ii,jj:dX+jj].sum()
            testSetf[i*teS2+j,:]=testImsi[i*teS2+j,::].flatten()

    return trainSetf, testSetf

def pca(trainSet, testSet, k):

    imsi=np.cov(trainSet.T)
    # L,V=np.linalg.eig(imsi)
    U,s,V=np.linalg.svd(imsi)
    #print(s)
    #plt.plot(s);plt.show()
    PC=U[:,np.argsort(s)[::-1]][:,:k]

    trainSetf=trainSet.dot(PC)
    testSetf=testSet.dot(PC)

    return trainSetf, testSetf

def lda(trainSet, testSet, k):
    trS1=len(trainSet)
    trS2=len(trainSet[0])
    trS3=int(trS1/10)
    
    covMat=np.zeros((10,trS2,trS2))
    meanV=np.zeros((10,trS2))

    for i in range(10):
        covMat[i,::]=np.cov(trainSet[i*trS3:(i+1)*trS3,:].T)
        meanV[i,:]=np.mean(trainSet[i*trS3:(i+1)*trS3,:],0)

    meanC=np.mean(meanV,0)
    Sb=(meanV-meanC).T.dot(meanV-meanC)
    Sw=covMat.sum(0)
    imsi=np.linalg.pinv(Sw).dot(Sb)
    # L,V=np.linalg.eig(imsi)
    U,s,V=np.linalg.svd(imsi)
    PC=U[:,np.argsort(s)[::-1]][:,:k]
    trainSetf=trainSet.dot(PC)
    testSetf=testSet.dot(PC)

    return trainSetf, testSetf


def nBayes(trainSet, testSet, case=5):

    trS1, trS2=trainSet.shape; trS3=int(trS1/10)
    teS1, teS2=testSet.shape; teS3=int(teS1/10)
    result=np.zeros((teS3,10)); meanV=np.zeros((10,trS2))
    covC=np.zeros((10,trS2,trS2)); g=np.zeros((10,teS1))

    for i in range(10):
        meanV[i,:]=np.mean(trainSet[i*trS3:(i+1)*trS3,:],axis=0)
        covC[i,::]=np.cov(trainSet[i*trS3:(i+1)*trS3,:].T)

    if case==3:
        covC=np.tile(np.mean(covC,axis=0),(10,1,1))
    if case==4:
        for i in range(10):
            covC[i,::]=np.diag(np.diag(covC[i,:]))

    for j in range(10):
        m=meanV[j,:]
        invCov=np.linalg.pinv(covC[j,::])
        covSum=np.diag(covC[j,::]).sum()
        for i in range(teS1):
            g[j,i]= -0.5* (testSet[i,:]-m).dot(invCov).dot(testSet[i,:]-m.T)-0.5*np.log(covSum)

#    pdb.set_trace()

    imsi = np.argmax(g, 0)
    for i in range(teS1):
        result[i%teS3,i//teS3]=imsi[i]
##
####    for i in range(teS1):
##        for j in range(10):
##            g[j]=-0.5*(testSet[i,:]-meanV[j,:]).dot(np.linalg.inv(covC[j,::])).dot((testSet[i,:]-meanV[j,:]).T)-0.5*np.log(np.diag(covC[j,::]).sum())
##        result[i%teS3,i//teS3]=np.argmax(g)

    return result
    
    
