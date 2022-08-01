#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import DetCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import random


# In[3]:


def extract2Ddata(fileName,delim = ','):
    f = open(fileName, "r")
    data = f.readlines()
    f.close()
    X_all = []
    Y_all = []
    for d in data:
        X1,X2,Y= map(float,d[:].split(delim))
        X_all.append([X1,X2])
        Y_all.append(Y)
    return X_all,Y_all


# In[4]:


X,Y = extract2Ddata(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\18-20220327T100954Z-001\18\train.txt")


# In[5]:


Xtrain = np.array(X).T
Class = np.array(Y)  
markers = ["." , "+"]


# In[84]:


plt.figure(figsize=(10,10))
for i in [1,2]:  
  plt.scatter(Xtrain[0,np.where(Class == i)],Xtrain[1,np.where(Class == i)],label = 'Actual class {}'.format(i), s = 50,marker = markers[i-1])
plt.scatter(kmeans[:,0],kmeans[:,1])
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Train data with actual classes')


# In[6]:


Xt = [Xtrain[:,np.where(Class == i)].reshape(2,-1).T for i in range(1,3)]


# In[211]:


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
    cov_diag = []
    for j in range(K):
        nk = len(np.where(points==j)[0])
        cov.append(sum([(data[i].reshape(d,1)-means[j].reshape(d,1))@(data[i].reshape(d,1)-means[j].reshape(d,1)).T for i in np.where(points==j)[0]])/nk)
        cov_diag.append(np.diag(np.diag(cov[j])))
        pik.append(nk/n)
    
    return np.array(means),np.array(cov),np.array(cov_diag),pik


# In[259]:


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
    temp_cov_diag = [np.diag(np.diag(temp_cov[i])) for i in range(K)]
    return np.array(temp_pik), np.array(temp_means), np.array(temp_cov),np.array(temp_cov_diag)


# In[213]:


#For diagonal matrices
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
    temp_cov_diag = [np.diag(np.diag(temp_cov[i])) for i in range(K)]
    return np.array(temp_pik), np.array(temp_means), np.array(temp_cov_diag)


# In[260]:


K = 13; d=2 ; niter =2

means_all = []; cov_all = []; pik_all = []; covdiag_all = []
for c in [0,1]:
    n = len(Xt[c])
    clusters_means,clusters_cov,clusters_covdiag,clusters_pik = Kmeans_update(Xt[c],K)
    for i in range(niter):
        clusters_pik,clusters_means,clusters_cov,clusters_covdiag = clusters_update(clusters_pik,clusters_means,clusters_cov)
        
    means_all.append(clusters_means)
    cov_all.append(clusters_cov)
    covdiag_all.append(clusters_covdiag)
    pik_all.append(clusters_pik)


# In[238]:


#for diagonal matrices
K = 18; d=2 ; niter = 2
means_all = []; cov_all = []; pik_all = []
for c in [0,1]:
    n = len(Xt[c])
    clusters_means,clusters_cov,clusters_covdiag,clusters_pik = Kmeans_update(Xt[c],K)
    for i in range(niter):
        clusters_pik,clusters_means,clusters_cov = clusters_update(clusters_pik,clusters_means,clusters_covdiag)
        
    means_all.append(clusters_means)
    cov_all.append(clusters_cov)
    pik_all.append(clusters_pik)


# In[104]:


def classification(X,pik,means,cov):
    Class = []
    for i in range(len(X)):
        g = {c+1:sum([pik[c][j]*np.exp(-0.5*(X[i].reshape(1,d)-means[c][j].reshape(1,d))@np.linalg.inv(cov[c][j])@(X[i].reshape(1,d)-means[c][j].reshape(1,d)).T)/(np.linalg.det(cov[c][j]))**0.5 for j in range(K)]) for c in range(2)}
        Class.append(max(g, key=g.get))
    return np.array(Class)  


# In[69]:


Xd,Yd = extract2Ddata(r"C:\Users\HEMA\OneDrive\Documents\PRML\Assignment_3\18-20220327T100954Z-001\18\dev.txt")


# In[70]:


Nd = len(Xd)
Xd = np.array(Xd).reshape(Nd,d)
Yd = np.array(Yd)


# In[239]:


Class_pre = classification(Xd,pik_all,means_all,cov_all)
Class_pre_diag = classification(Xd,pik_all,means_all,covdiag_all)


# In[267]:


plt.figure(figsize=(10,10))
for i in [1,2]:  
  plt.scatter(Xd[np.where(Class_pre == i),0],Xd[np.where(Class_pre == i),1],label = 'predicted class {}'.format(i), s = 50,marker = markers[i-1])
plt.legend()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Train data with actual classes')


# In[207]:


#confusion matrix taking full normal matrices
Cdev = confusion_matrix(Yd, Class_pre)
ConfusionMatrixDisplay(confusion_matrix=Cdev,display_labels=['Class 1','Class 2']).plot()
plt.title('Confusion Matrix:(DEV) taking Full Matrix')


# In[240]:


#confusion matrix taking diagonal normal matrices
Cdev = confusion_matrix(Yd, Class_pre_diag)
ConfusionMatrixDisplay(confusion_matrix=Cdev,display_labels=['Class 1','Class 2']).plot()
plt.title('Confusion Matrix:(DEV) taking diagonal Matrix')


# In[277]:


#preparing meshgrid for plots
import numpy as np
nplot = 100
d = 2
f = 0.6
Xmin = np.min(Xtrain[0,:]); Xmax = np.max(Xtrain[0,:]); 
Ymin = np.min(Xtrain[1,:]); Ymax = np.max(Xtrain[1,:]);
Dx = Xmax-Xmin; Dy = Ymax-Ymin; 
MidX = (Xmin+Xmax)/2; MidY = (Ymin+Ymax)/2;
Xmin = MidX-f*Dx; Xmax = MidX+f*Dx;
Ymin = MidY-f*Dy; Ymax = MidY+f*Dy;
xp = np.linspace(Xmin,Xmax,nplot)
yp = np.linspace(Ymin,Ymax,nplot)
Xp,Yp = np.meshgrid(xp,yp)


# In[278]:


#plotting decision boundary
Z = np.zeros((nplot,nplot))
S = {i:np.zeros((nplot,nplot)) for i in [1,2]}
for i in range(nplot):
  for j in range(nplot):
    x = np.array([Xp[i,j],Yp[i,j]])
    g = {c+1:np.sum([pik_all[c][a]*np.exp(-0.5*(x.T-means_all[c][a].reshape(1,d))@np.linalg.inv(cov_all[c][a])@(x-means_all[c][a].reshape(d,1)))/(np.linalg.det(cov_all[c][a]))**0.5 for a in range(K)]) for c in range(2)}
    #print(type(g),g)
    Z[i,j] = max(g,key=g.get)
    for c in [1,2]:
      S[c][i,j] = np.log(np.sum([pik_all[c-1][a]*np.exp(-0.5*(x.T-means_all[c-1][a].reshape(1,d))@np.linalg.inv(cov_all[c-1][a])@(x-means_all[c-1][a].reshape(d,1)))/(np.linalg.det(cov_all[c-1][a]))**0.5 for a in range(K)]))    


# In[279]:


#plotting contours
plt.figure(figsize=(10,10))
plt.pcolormesh(Xp,Yp,Z)
for i in [1,2]:  
  plt.scatter(Xtrain[0,np.where(Class == i)],Xtrain[1,np.where(Class == i)],label = 'Actual class {}'.format(i), s = 50,marker = markers[i-1])
  plt.contour(Xp,Yp, S[i])
plt.legend()

plt.xlabel('X1')
plt.ylabel('Y1')
plt.title('Case 2 Prediction over test data', fontsize=20)  

plt.show()