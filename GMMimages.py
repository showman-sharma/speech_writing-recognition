#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random

def extract_data(directory):
    Xt_class = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            data = f.readlines()
            f.close()
            for d in data:
                Xt_class.append(np.array(list(map(float,d[:].split( )))))
        
    return np.array(Xt_class)
    


# In[45]:


Xt = [0]*5
Xt[0] = extract_data(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\coast\train")
Xt[1] = extract_data(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\forest\train")
Xt[2] = extract_data(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\highway\train")
Xt[3] = extract_data(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\mountain\train")
Xt[4] = extract_data(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\opencountry\train")


# In[24]:


import os
import numpy as np

def extract_data_dev(directory):
    Xd_class = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            temp = []
            data = f.readlines()
            f.close()
            for d in data:
                temp.append(np.array(list(map(float,d[:].split( )))))
            Xd_class.append(np.array(temp))
        
    return (Xd_class)


# In[46]:


Xd = []; numd= []
class_actual = []
Xd_class = extract_data_dev(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\coast\dev")
Xd.extend(Xd_class)
class_actual.extend([1]*len(Xd_class))
numd.append(len(Xd_class))
Xd_class = extract_data_dev(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\forest\dev")
Xd.extend(Xd_class)
class_actual.extend([2]*len(Xd_class))
numd.append(len(Xd_class))
Xd_class = extract_data_dev(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\highway\dev")
Xd.extend(Xd_class)
class_actual.extend([3]*len(Xd_class))
numd.append(len(Xd_class))
Xd_class = extract_data_dev(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\mountain\dev")
Xd.extend(Xd_class)
class_actual.extend([4]*len(Xd_class))
numd.append(len(Xd_class))
Xd_class = extract_data_dev(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\Features\opencountry\dev")
Xd.extend(Xd_class)
class_actual.extend([5]*len(Xd_class))
numd.append(len(Xd_class))
Xd = np.array(Xd)


# In[40]:


d = 23
for c in range(5):
    for j in range(d):
        mini = min(Xt[c][:,j])
        maxi = max(Xt[c][:,j])
        diff = maxi - mini
        Xt[c][:,j] = (Xt[c][:,j]-mini)*100/diff
        Xd[:numd[c],:,j] = (Xd[:numd[c],:,j]-mini)*100/diff
    


# In[11]:


def Kmeans_update(data,K,Niter = 10):
    indices = np.random.choice(data.shape[0],K)
    means = data[indices,:]
    distances = cdist(data,means)
    points = np.array([np.argmin(d) for d in distances])
    
    for _ in range(Niter):
        means = [sum([data[i] for i in np.where(points==j)[0]])/len(np.where(points==j)[0]) for j in range(K)]
        distances = cdist(data,means)
        points = np.array([np.argmin(d) for d in distances])
    cov = []
    pik = []
    for j in range(K):
        nk = len(np.where(points==j)[0])
        cov.append(sum([(data[i].reshape(d,1)-means[j].reshape(d,1))@(data[i].reshape(d,1)-means[j].reshape(d,1)).T for i in np.where(points==j)[0]])/nk)
        pik.append(nk/n)
    
    return np.array(means),np.array(cov),pik


# In[12]:


def clusters_update(pik,means,cov):
    gamma = np.zeros((n,K))
    for i in range(n):
        den = 0 
        for j in range(K):
            
            t1 = (-0.5*(Xt[c][i].reshape(1,d)-means[j].reshape(1,d))@np.linalg.inv(cov[j])@(Xt[c][i].reshape(d,1)-means[j].reshape(d,1)))
            den += pik[j]*np.exp(t1)/(np.linalg.det(cov[j]))**0.5
        
        for j in range(K):
            num = pik[j]*np.exp(-0.5*(Xt[c][i].reshape(1,d)-means[j].reshape(1,d))@np.linalg.inv(cov[j])@(Xt[c][i].reshape(d,1)-means[j].reshape(d,1)))/(np.linalg.det(cov[j]))**0.5
            gamma[i,j] = num/den
    
    nk = [sum([gamma[i,j] for i in range(n)]) for j in range(K)]
    temp_means = [sum([gamma[i,j]*Xt[c][i] for i in range(n)])/nk[j] for j in range(K)]
    temp_cov = [sum([gamma[i,j]*(Xt[c][i].reshape(d,1)-means[j].reshape(d,1))@(Xt[c][i].reshape(d,1)-means[j].reshape(d,1)).T for i in range(n)])/nk[j] for j in range(K)] 
    temp_pik = [nk[j]/n for j in range(K)]
    print(sum(temp_pik))
    return np.array(temp_pik), np.array(temp_means), np.array(temp_cov)


# In[59]:


k = 20; d=23 ; niter =7
means_all = []; cov_all = []; pik_all = []
for c in [0,1,2,3,4]:
    n = len(Xt[c])
    clusters_means,clusters_cov,clusters_pik = Kmeans_update(Xt[c],K)
    for i in range(niter):
        clusters_pik,clusters_means,clusters_cov = clusters_update(clusters_pik,clusters_means,clusters_cov)
        
    means_all.append(clusters_means)
    cov_all.append(clusters_cov)
    pik_all.append(clusters_pik)


# In[65]:



def classification(X,pik,means,cov):
    Class = []
    b = 36
    ans = -1
    temp1 = 0
    for i in range(len(X)):
        for c in range(5):
            temp2 = 0
            for a in range(b):
                temp2 += np.log(sum([pik[c][j]*np.exp(-0.5*(X[i][a]-means[c][j]).T@np.linalg.pinv(cov[c][j])@(X[i][a]-means[c][j]))/(np.linalg.det(cov[c][j]))**0.5 for j in range(K)]))
            #print(temp2)
            if temp2>temp1:
                temp1 = temp2 
                ans = c
        Class.append(ans+1)
        temp1 = 0
    return np.array(Class) 


# In[69]:


Cdev = confusion_matrix(class_actual, class_pre)
ConfusionMatrixDisplay(confusion_matrix=Cdev,display_labels=['Coast','forest','highway','mountain','open country']).plot()
plt.title('Confusion Matrix:(DEV)')

plt.show()