#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import DetCurveDisplay,ConfusionMatrixDisplay, confusion_matrix
import sys

try:
    dirAssign = sys.argv[1]#"/mnt/d/SEM VIII/PRML/assignments/Assignment 3"
except:
    dirAssign = os.getcwd()
    
print('HMM Code for Writing Recognition Begins')

def Kmeans(data,K,Niter = 10):
    indices = np.random.choice(data.shape[0],K)
    means = data[indices,:]
    distances = cdist(data,means)
    points = np.array([np.argmin(d) for d in distances])
    
    for _ in range(Niter):
        means = np.array([data[points==i].mean(axis=0) for i in range(K)])
        distances = cdist(data,means)
        points = np.array([np.argmin(d) for d in distances])
    return points,means     


# In[8]:


classes = ['a', 'ai', 'bA', 'lA', 'tA']


# # Reading data

# In[9]:




# # TELUGU CHARACTERS

# Character: a, ai, bA, lA, tA

# In[10]:


def readPoints(fileName):
    f = open(fileName, "r")
    file = []
    for x in f:
        file.append(x)
    data = list(map(float,(file[0]).split()))  
    NC = int(data[0])
    coords = []
    for i in range(NC):
        coords.append([data[i*2+1],data[2*i+2]])
    return np.array(coords).T


# ## Extracting train and dev data

# In[11]:


def meanShift(points):
    newPoints = points-np.mean(points,axis = 1).reshape(-1,1)
    return newPoints


# In[12]:


classes = ['a', 'ai', 'bA', 'lA', 'tA']


# In[13]:


train = {i:[] for i in classes}
dev = {i:[] for i in classes}
for c in classes:
    dirT = dirAssign+"/"+str(c)+"/train"
    dirD = dirAssign+"/"+str(c)+"/dev"
    for filename in os.listdir(dirT):
        train[c].append(readPoints(str(os.path.join(dirT, filename)).replace('\\','/')))
        
    for filename in os.listdir(dirD):
        dev[c].append(readPoints(str(os.path.join(dirD, filename)).replace('\\','/')))
               


# In[14]:


for c in classes:
    train[c] = [meanShift(t) for t in train[c]]
    dev[c] = [meanShift(d) for d in dev[c]]


# # Encoding

# In[15]:


#preparring big bag of raw unsorted unclassified data for creating clusters
def dataBag(train):
    data = np.zeros((0,train[classes[0]][0].shape[0]))
    for c in train:
        for t in train[c]:
            data = np.append(data,t.T,axis=0)
    return data        


# In[16]:


data = dataBag(train)


# In[17]:


K = 12


# In[18]:


points,means = Kmeans(data,K)


# In[19]:


def Encode(data,means):
    distances = cdist(data.T,means)
    points = np.array([np.argmin(d) for d in distances])
    return points


# In[20]:


def CodeBook(data,means):
    classes = list(data.keys())
    encoded = {i:[] for i in classes}
    for c in classes:
        for d in data[c]:
            encoded[c].append(Encode(d,means))
    return encoded


# In[21]:


encodedT = CodeBook(train,means)


# In[22]:


encodedD = CodeBook(dev,means)


# # Finding number of states

# In[23]:


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


# In[24]:


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


# In[25]:


'''
start = time.time()  
trainMeds = {i:Medoid(train[i],dtw) for i in classes}
end = time.time()    
print('Medoid finding runtime = {} seconds'.format(end-start))  
'''


# In[26]:


'''
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(5,figsize=(10,10))
  
# For Sine Function
for i in range(5):
    axis[i].plot(trainMeds[classes[i]])
    axis[i].set_title('Class = {}'.format(classes[i]))
  
  
# Combine all the operations and display
plt.show()
'''


# Looking at the above plot, it looks like
# - 2 -> 3 states
# - 3 -> 5 states
# - 4 -> 5 states
# - 8 -> 4 states
# - 9 -> 5 states

# In[27]:


for cl in classes:
    ex = 0
    plt.figure(figsize=(5,5))
    plt.plot(train[cl][ex][0,:], train[cl][ex][1,:])
    plt.title('Letter '+cl)


# In[28]:


stateCount = {'a':5, 'ai':7, 'bA':7, 'lA':6, 'tA':7}


# # Training HMM parameters

# In[29]:


def trainHMM(encodedT,classes,stateCount,K,dirName = dirAssign+"/HMM-Code"):
    
    A = {} #transition probabilities
    B = {} #symbol probabs when in transitioning to same state
    C = {} #symbol probabs when in transitioning to next state
    
    os.chdir(dirName)
    for c in classes:
        #Creating HMM model for the class
        with open('test.hmm.seq','w') as file:
            for example in (encodedT[c]):
                for letter in example:
                    file.write(str(letter)+' ')
                file.write('\n')    
        os.system(f"./train_hmm test.hmm.seq 1234 {stateCount[c]} {K} .01")
        with open('test.hmm.seq.hmm','r') as file:
            lines = file.readlines()
            #print(lines)
            l = len(lines)-2
            A[c] = np.eye(stateCount[c])
            B[c] = np.zeros((stateCount[c],K))
            C[c] = np.zeros((stateCount[c],K))

            for i in range(l//3):
                line1 = list(map(float,lines[3*i+2].split()))
                line2 = list(map(float,lines[3*i+3].split()))
                #print(lines[2*i-1])
                #print(lines[3*i+3])
                A[c][i,i] = line1[0];
                B[c][i,:] = line1[1:]
                #try:
                #print(line2[0])
                A[c][i,i+1] = line2[0]
                C[c][i,:] = line2[1:]
                #except:
                    #print('lol')
    return A,B,C


# In[30]:


def trainHMM(encodedT,classes,stateCount,K,dirName = "/mnt/d/SEM VIII/PRML/assignments/Assignment 3/HMM-Code"):
    
    A = {} #transition probabilities
    B = {} #symbol probabs when in transitioning to same state
    C = {} #symbol probabs when in transitioning to next state
    recP = {c:[] for c in classes}
    traP = {c:[] for c in classes}
    
    os.chdir(dirName)
    for c in classes:
        #Creating HMM model for the class
        with open('test.hmm.seq','w') as file:
            for example in (encodedT[c]):
                for letter in example:
                    file.write(str(letter)+' ')
                file.write('\n')    
        os.system(f"./train_hmm test.hmm.seq 1234 {stateCount[c]} {K} .01")
        with open('test.hmm.seq.hmm','r') as file:
            lines = file.readlines()
            #print(lines)
            l = len(lines)-2
            A[c] = np.eye(stateCount[c])
            B[c] = np.zeros((stateCount[c],K))
            C[c] = np.zeros((stateCount[c],K))
            print(stateCount[c],l//3)
            for i in range(l//3):
                line1 = list(map(float,lines[3*i+2].split()))
                line2 = list(map(float,lines[3*i+3].split()))

                A[c][i,i] = line1[0];
                B[c][i,:] = line1[1:]
                try:
                    A[c][i,i+1] = line2[0]
                except:
                    pass
                C[c][i,:] = line2[1:]
                recP[c].append(line1[0])
                traP[c].append(line2[0])
            recP[c] = np.array(recP[c])
            traP[c] = np.array(traP[c])
    return A,B,C,recP,traP


# In[31]:


A,B,C,recP,traP = trainHMM(encodedT,classes,stateCount,K)


# # Testing Dev Data

# ## Classifier 1

# In[32]:


def forward1(O,recP,traP,B,C):
    N,M = B.shape #  N=#states, M=#symbos
    T = O.shape[0]
    alpha = np.zeros((T+1,N))
    alpha[0, 0] = 1 
    for t in range(T):
        for j in range(N):
            alpha[t+1, j] = alpha[t,j]*recP[j]*B[j][O[t]]
            if j>0:
                alpha[t+1, j] += alpha[t,j-1]*traP[j-1]*C[j-1][O[t]]  
    return np.sum(alpha[-1,:])


# In[33]:


def classifier1(O,recP,traP,B,C,classes):
    prob = {i:0 for i in classes}
    for c in classes:
        #dists = [dtw(tr,dev,window) for tr in train[c]]
        prob[c] = np.log(forward1(O,recP[c],traP[c],B[c],C[c]))/O.shape[0]#np.mean(dists)
    return max(prob, key = lambda c: prob[c]),prob


# In[34]:


def test1(encodedD,recP,traP,B,C,K):
    start = time.time()  
    classes = recP.keys()
    print(classes)
    pred = {i:[] for i in classes}
    probs = []
    for c in classes:
        print(c)
        print(encodedD[c])
        for d in encodedD[c]:
            p,prob = classifier1(d,recP,traP,B,C,classes)
            pred[c].append(p)
            probs.append(prob)
        #pred2[c] = [classifier2(d,trainMeds,classes) for d in dev[c]]
        #Dscore2 = Dscore2 + []
    end = time.time()    
    print('Classifier 1 with K = {} runtime = {} seconds'.format(K,end-start))
    return pred,probs


# In[35]:


pred1,probs1 = test1(encodedD,recP,traP,B,C,K)


# # Measure performance

# In[36]:


y_true = []
for c in classes:
    y_true = y_true + [c]*len(dev[c])


# In[37]:


def ROC_DET(Y,prob,classes,N=1000):
    Smin = min([min(p.values()) for p in prob])#0#min([min(Dscores[:][c]) for c in classes])
    Smax = max([max(p.values()) for p in prob]) 
    threshs = np.array([])
    for p in prob:
        threshs = np.append(threshs,list(p.values()))
    threshs.sort()
    #print(threshs)    
    TPR = []; FPR = [];FNR = []
    for thresh in threshs:#np.linspace(Smin,Smax,N):
        TP = FN = TN = FP = 0;
        for i,y in zip(range(len(Y)),Y):
            for c in classes:
                if prob[i][c] >= thresh:
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


# In[38]:


TPR1,FPR1,FNR1 = ROC_DET(y_true,probs1,classes)


# In[39]:


plt.figure(figsize=(10,10))
plt.plot(FPR1,TPR1,label = 'classifier 1')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('Telugu Text: ROC for Classifiers ')


# In[40]:


fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
DetCurveDisplay(fpr=FPR1, fnr=FNR1, estimator_name="Classifier 1").plot(ax = ax_det);
ax_det.set_title('Telugu Text: DET curve for classifiers')


# In[41]:


y_pred1 = [];
for c in classes:
    y_pred1 = y_pred1 + pred1[c]
Cdev1 = confusion_matrix(y_true, y_pred1)


# In[42]:


ConfusionMatrixDisplay(confusion_matrix=Cdev1,display_labels=classes).plot()
plt.title('Telugu Text: Confusion Matrix for classifier 1')
print('accuracy = {}%'.format(100*np.trace(Cdev1)/np.sum(Cdev1)))


# # Testing for different K values

# In[43]:


Ks = [5,10,15,20,25,30,35]
TPR = {};FPR = {};FNR = {};
Cdev = {}
pred = {};probs = {}
At = {};Bt = {};Ct={};recPt={};traPt={}
encodedT = {}; encodedD = {}
for K in Ks:
    points,means = Kmeans(data,K)
    encodedT[K] = CodeBook(train,means)
    encodedD[K] = CodeBook(dev,means)
    print(np.unique(encodedT[K][classes[0]][0]))
    At[K],Bt[K],Ct[K],recPt[K],traPt[K] = trainHMM(encodedT[K],classes,stateCount,K)
    pred[K],probs[K] = test1(encodedD[K],recPt[K],traPt[K],Bt[K],Ct[K],K)
    TPR[K],FPR[K],FNR[K] = ROC_DET(y_true,probs[K],classes)
    y_pred = []
    for c in classes:
        y_pred = y_pred + pred[K][c]
    Cdev[K] = confusion_matrix(y_true, y_pred)
    


# In[44]:


plt.figure(figsize=(10,10))
for K in Ks:
    plt.plot(FPR[K],TPR[K],label = 'Classifier with K = {}'.format(K))
#plt.plot(FPR5,TPR5,label = 'Classifier 5')
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),linestyle='--', label = 'Random prediction')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Telugu Text: ROC curve for different HMM classifiers')
plt.legend()

fig, ax_det = plt.subplots(1,1,figsize=(10, 10))
for K in Ks:
    DetCurveDisplay(fpr=FPR[K], fnr=FNR[K], estimator_name="Classifier with K = {}".format(K)).plot(ax = ax_det)
#DetCurveDisplay(fpr=FPR5, fnr=FNR5, estimator_name="Classifier 5").plot(ax = ax_det)
ax_det.set_title('Telugu Text: DET curve for different HMM classifiers')


# In[45]:


for K in Ks:
    print('accuracy@(K={})= {}%'.format(K,100*np.trace(Cdev[K])/np.sum(Cdev[K])))


# In[ ]:



plt.show()

# In[ ]:




