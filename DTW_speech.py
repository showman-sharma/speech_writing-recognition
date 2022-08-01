

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import DetCurveDisplay,ConfusionMatrixDisplay, confusion_matrix
import sys

try:
    dirAssign = sys.argv[1]#"/mnt/d/SEM VIII/PRML/assignments/Assignment 3"
except:
    dirAssign = os.getcwd()

# In[6]:

print('DTW Code for Speech Recognition Begins')
# https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
def dtw(s0, t0, window = 10):
    s = s0/np.max(s0,axis=1).reshape(-1,1)
    t = t0/np.max(t0,axis=1).reshape(-1,1)
    n, m = s.shape[1], t.shape[1]
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = np.linalg.norm(s[:,i-1] - t[:,j-1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[-1,-1]


# # SPOKEN DIGITS

# Digits: 2,3,4,8,9

# In[7]:


def readMFCC(fileName):
    f = open(fileName, "r")
    file = []
    for x in f:
        file.append(x)
    NC,NF = map(int,(file[0]).split())  
    data = np.zeros((NC,NF))
    for i in range(len(file)-1):
        data[:,i] = np.array(list(map(float,(file[i+1]).split())))
        
    return data


# ## Extracting train and dev data

# In[8]:


classes = [2,3,4,8,9]


# In[9]:



train = {i:[] for i in classes}
dev = {i:[] for i in classes}
for c in classes:
    dirT = dirAssign +"/"+str(c)+"/train"
    dirD = dirAssign+"/"+str(c)+"/dev"
    for filename in os.listdir(dirT):
        if filename.endswith(".mfcc"): 
            train[c].append(readMFCC(str(os.path.join(dirT, filename)).replace('\\','/')))
        else:
            continue
    for filename in os.listdir(dirD):
        if filename.endswith(".mfcc"): 
            dev[c].append(readMFCC(str(os.path.join(dirD, filename)).replace('\\','/')))
        else:
            continue        


# In[10]:



# ## Testing DTW




y_true = []
for c in classes:
    y_true = y_true + [c]*len(dev[c])


# In[15]:


def ROC_DET(Y,Dscores,classes,N=1000):
    Smin = min([min(D.values()) for D in Dscores])#0#min([min(Dscores[:][c]) for c in classes])
    Smax = max([max(D.values()) for D in Dscores]) 
    TPR = []; FPR = [];FNR = []
    for thresh in np.linspace(Smin,Smax,N):
        TP = FN = TN = FP = 0;
        for i,y in zip(range(len(Y)),Y):
            for c in classes:
                if Dscores[i][c] <= thresh:
                    if y==c:
                        TP+=1
                    else:
                        FP+=1
                else:
                    if y==c:
                        FN+=1
                    else:
                        TN+=1
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(FP+TN))
        FNR.append(FN/(TP+FN))
    return TPR,FPR,FNR


# ### Classifier 1

# In[16]:


def classifier1(dev,train,classes,window = 10):
    avgDist = {i:0 for i in classes}
    for c in classes:
        dists = [dtw(tr,dev,window) for tr in train[c]]
        avgDist[c] = np.mean(dists)/dev.shape[1]
    return min(avgDist, key = lambda c: avgDist[c]), avgDist   


# In[17]:


start = time.time()  
pred1 = {i:[] for i in classes}
Dscore1 = []
for c in classes:
    for d in dev[c]:
        p,D = classifier1(d,train,classes)
        pred1[c].append(p)
        Dscore1.append(D)
    #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
    #Dscore2 = Dscore2 + []
end = time.time()    
print('Classifier 1 runtime = {} seconds'.format(end-start))


# In[18]:


TPR1,FPR1,FNR1 = ROC_DET(y_true,Dscore1,classes)


# In[31]:


plt.figure(figsize=(10,10))
plt.plot(FPR1,TPR1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC for DTW Classifier 1')


# In[32]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR1, fnr=FNR1, estimator_name="Classifier 1").plot(ax = ax_det);
ax_det.set_title('Spoken Digit: DET curve for DTW classifier 1')


# In[33]:


y_pred1 = []
for c in classes:
    y_pred1 = y_pred1 + pred1[c]
Cdev1 = confusion_matrix(y_true, y_pred1)

ConfusionMatrixDisplay(confusion_matrix=Cdev1,display_labels=classes).plot()
plt.title('Spoken Digit: Confusion Matrix for DTW classifier 1')


# Accuracy = 96.667%

# ### Classifier 2

# In[22]:


def Medoid(data,Dfunc):
    l = len(data)
    D = np.zeros((l,l))
    for i in range(l):
        for j in range(i+1,l):
            D[i,j] = Dfunc(data[i],data[j])
    #print(D)  
    d = [sum(D[k,:])+sum(D[:,k]) for k in range(l)]
    medoid = data[d.index(min(d))]
    #medoid = min(data, key = lambda k: sum(D[k,:])+sum(D[:,k]))
    return medoid    


# In[23]:


start = time.time()  
trainMeds = {i:Medoid(train[i],dtw) for i in classes}
end = time.time()    
print('Medoid finding runtime = {} seconds'.format(end-start))   


# In[24]:


def classifier2(dev,trainMeds,classes,window = 10):
    Dist = {i:0 for i in classes}
    for c in classes:
        #dists = [dtw(tr,dev,window) for tr in train[c]]
        Dist[c] = dtw(trainMeds[c],dev,window)/dev.shape[1]#np.mean(dists)
    return min(Dist, key = lambda c: Dist[c]),Dist    


# In[25]:


start = time.time()  
pred2 = {i:[] for i in classes}
Dscore2 = []
for c in classes:
    for d in dev[c]:
        p,D = classifier2(d,trainMeds,classes)
        pred2[c].append(p)
        Dscore2.append(D)
    #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
    #Dscore2 = Dscore2 + []
end = time.time()    
print('Classifier 2 runtime = {} seconds'.format(end-start))  


# In[26]:


TPR2,FPR2,FNR2 = ROC_DET(y_true,Dscore2,classes)


# In[27]:


plt.figure(figsize=(10,10))
plt.plot(FPR2,TPR2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC for DTW Classifier 2')


# In[34]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR2, fnr=FNR2, estimator_name="Classifier 2").plot(ax = ax_det);
ax_det.set_title('Spoken Digit: DET curve for DTW classifier 2')


# In[35]:


y_pred2 = []
for c in classes:
    y_pred2 = y_pred2 + pred2[c]
Cdev2 = confusion_matrix(y_true, y_pred2)
ConfusionMatrixDisplay(confusion_matrix=Cdev2,display_labels=classes).plot()
plt.title('Spoken Digit: Confusion Matrix for DTW classifier 2')


# Accuracy = 95%

# ### Classifier 3
# Mean shifted to midpoint of each hand written letter. Mean shifts of medoids of training data were used.

# In[36]:


def meanShift(points):
    newPoints = points-np.mean(points,axis = 1).reshape(-1,1)
    return newPoints


# In[38]:


def classifier3(dev,trainMeds,classes,window = 10):
    shiftedMeds = {c:meanShift(trainMeds[c]) for c in classes}
    Dist = {i:0 for i in classes}
    for c in classes:
        #dists = [dtw(tr,dev,window) for tr in train[c]]
        Dist[c] = dtw(shiftedMeds[c],meanShift(dev),window)/dev.shape[1]#np.mean(dists)
    return min(Dist, key = lambda c: Dist[c]),Dist 


# In[39]:


start = time.time()  
pred3 = {i:[] for i in classes}
Dscore3 = []
for c in classes:
    for d in dev[c]:
        p,D = classifier3(d,trainMeds,classes)
        pred3[c].append(p)
        Dscore3.append(D)
    #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
    #Dscore2 = Dscore2 + []
end = time.time()    
print('Classifier 3 runtime = {} seconds'.format(end-start)) 


# In[40]:


TPR3,FPR3,FNR3 = ROC_DET(y_true,Dscore3,classes)


# In[41]:


plt.figure(figsize=(10,10))
plt.plot(FPR3,TPR3)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC for DTW Classifier 3')


# In[42]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR3, fnr=FNR3, estimator_name="Classifier 3").plot(ax = ax_det);
ax_det.set_title('Spoken Digit: DET curve for DTW classifier 3')


# In[43]:


y_pred3 = []
for c in classes:
    y_pred3 = y_pred3 + pred3[c]
Cdev3 = confusion_matrix(y_true, y_pred3)

ConfusionMatrixDisplay(confusion_matrix=Cdev3,display_labels=classes).plot()
plt.title('Spoken Digit: Confusion Matrix for DTW classifier 3')


# Accuracy = 95%

# ### Classifier 4
# Medoid of mean shifted training data is used

# In[46]:


def classifier4(dev,trainMedsShifted,classes,window = 10):
    #shiftedMeds = {c:meanShift(trainMeds[c]) for c in classes}
    Dist = {i:0 for i in classes}
    for c in classes:
        #dists = [dtw(tr,dev,window) for tr in train[c]]
        Dist[c] = dtw(trainMedsShifted[c],meanShift(dev),window)/dev.shape[1]#np.mean(dists)
    return min(Dist, key = lambda c: Dist[c]),Dist 


# In[47]:


start = time.time()  
trainMedsShifted = {i:Medoid([meanShift(t) for t in train[i]],dtw) for i in classes}
end = time.time()    
print('Medoid of mean shifted data finding runtime = {} seconds'.format(end-start))   


# In[48]:


start = time.time()  
pred4 = {i:[] for i in classes}
Dscore4 = []
for c in classes:
    for d in dev[c]:
        p,D = classifier4(d,trainMedsShifted,classes)
        pred4[c].append(p)
        Dscore4.append(D)
    #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
    #Dscore2 = Dscore2 + []
end = time.time()    
print('Classifier 4 runtime = {} seconds'.format(end-start))  


# In[49]:


TPR4,FPR4,FNR4 = ROC_DET(y_true,Dscore4,classes)


# In[50]:


plt.figure(figsize=(10,10))
plt.plot(FPR4,TPR4)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC for DTW Classifier 4')


# In[51]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR4, fnr=FNR4, estimator_name="Classifier 4").plot(ax = ax_det);
ax_det.set_title('Spoken Digit: DET curve for DTW classifier 4')


# In[52]:


y_pred4 = []
for c in classes:
    y_pred4 = y_pred4 + pred4[c]
Cdev4 = confusion_matrix(y_true, y_pred4)

ConfusionMatrixDisplay(confusion_matrix=Cdev4,display_labels=classes).plot()
plt.title('Spoken Digit: Confusion Matrix for DTW classifier 4')


# ### Classifier 5
# Normalization of mean shifted versions of data. Medoids were used after normalization of mean shifts of training data.

# In[54]:


def normalize(points):
    return points/np.max(points,axis = 1).reshape(-1,1)


# In[55]:


def classifier5(dev,trainMedsShiftedNormalized,classes,window = 10):
    #shiftedMeds = {c:meanShift(trainMeds[c]) for c in classes}
    Dist = {i:0 for i in classes}
    for c in classes:
        #dists = [dtw(tr,dev,window) for tr in train[c]]
        #t = normalize(trainMedsShiftedNormalized[c])#/np.max(trainMedsShifted[c],axis = 1).reshape(-1,1)
        s = normalize(meanShift(dev))#/np.max(meanShift(dev),axis = 1).reshape(-1,1)
        Dist[c] = dtw(trainMedsShiftedNormalized[c],s,window)/dev.shape[1]#np.mean(dists)
    return min(Dist, key = lambda c: Dist[c]),Dist 


# In[56]:


start = time.time()  
trainMedsShiftedNormalized = {i:Medoid([normalize(meanShift(t)) for t in train[i]],dtw) for i in classes}
end = time.time()    
print('Medoid of mean shifted normalized data finding runtime = {} seconds'.format(end-start))   


# In[57]:


start = time.time()  
pred5 = {i:[] for i in classes}
Dscore5 = []
for c in classes:
    for d in dev[c]:
        p,D = classifier5(d,trainMedsShiftedNormalized,classes)
        pred5[c].append(p)
        Dscore5.append(D)
    #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
    #Dscore2 = Dscore2 + []
end = time.time()    
print('Classifier 5 runtime = {} seconds'.format(end-start))  


# In[58]:


TPR5,FPR5,FNR5 = ROC_DET(y_true,Dscore5,classes)


# In[59]:


plt.figure(figsize=(10,10))
plt.plot(FPR5,TPR5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC for DTW Classifier 4')


# In[60]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR5, fnr=FNR5, estimator_name="Classifier 5").plot(ax = ax_det);
ax_det.set_title('Spoken Digit: DET curve for DTW classifier 5')


# In[61]:


y_pred5 = []
for c in classes:
    y_pred5 = y_pred5 + pred5[c]
Cdev5 = confusion_matrix(y_true, y_pred5)

ConfusionMatrixDisplay(confusion_matrix=Cdev5,display_labels=classes).plot()
plt.title('Spoken Digit: Confusion Matrix for DTW classifier 5')


# Accuracy = 98%

# ## Comparing models

# In[62]:


plt.figure(figsize=(10,10))
plt.plot(FPR1,TPR1,label = 'Classifier 1')
plt.plot(FPR2,TPR2,label = 'Classifier 2')
plt.plot(FPR3,TPR3,label = 'Classifier 3')
plt.plot(FPR4,TPR4,label = 'Classifier 4')
plt.plot(FPR5,TPR5,label = 'Classifier 5')
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle='--', label = 'Random prediction')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Spoken Digit: ROC curve for different DTW classifiers')
plt.legend()

fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR1, fnr=FNR1, estimator_name="Classifier 1").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR2, fnr=FNR2, estimator_name="Classifier 2").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR3, fnr=FNR3, estimator_name="Classifier 3").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR4, fnr=FNR4, estimator_name="Classifier 4").plot(ax = ax_det)
DetCurveDisplay(fpr=FPR5, fnr=FNR5, estimator_name="Classifier 5").plot(ax = ax_det)
ax_det.set_title('Spoken Digit: DET curve for different DTW classifiers')


# In[ ]:

plt.title('Spoken Digit Confusion Matrix for DTW classifiers')
print('DWT accuracy Classifier 1 = {}%'.format(100*np.trace(Cdev1)/np.sum(Cdev1)))
print('DWT accuracy Classifier 2 = {}%'.format(100*np.trace(Cdev2)/np.sum(Cdev2)))
print('DWT accuracy Classifier 3 = {}%'.format(100*np.trace(Cdev3)/np.sum(Cdev3)))
print('DWT accuracy Classifier 4 = {}%'.format(100*np.trace(Cdev4)/np.sum(Cdev4)))
print('DWT accuracy Classifier 5 = {}%'.format(100*np.trace(Cdev5)/np.sum(Cdev5)))


plt.show()