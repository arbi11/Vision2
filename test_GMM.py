# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 01:41:04 2018

@author: Arbi
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float

img = plt.imread('69020.jpg')
img = img_as_float(img)
n, d, r = img.shape

X = img.reshape([n*d, r])
n, d = X.shape
    
c = 2 ## number of clusters
eps = 0.001   
max_iters = 5 
# randomly choose the starting centroids/means 
## as 3 of the points from datasets        
mu = X[np.random.choice(n, c, False), :]

# initialize the covariance matrices for each gaussians
Sigma= [np.eye(d)] * c

# initialize the probabilities/weights for each gaussians
w = [1./ c] * c

# responsibility matrix is initialized to all zeros
# we have responsibility for each of n points for eack of k gaussians
R = np.zeros((n, c))

### log_likelihoods
log_likelihoods = []

P = lambda mu, s: (np.linalg.det(s) * ((2 * np.pi) ** (-X.shape[1]/2.)) ** (-.5)) \
* np.exp(-.5 * np.einsum('ij, ij -> i',\
        X - mu, np.dot(np.linalg.inv(s) , (X - mu).T).T)) 
        
# Iterate till max_iters iterations        
while len(log_likelihoods) < max_iters:

# E - Step
#s = Sigma[k]
#x_mu = X - mu[k]
#a = np.dot(np.linalg.inv(s) , (X - mu[k]).T).T
#a2 = np.einsum('ij, ij -> i',\
#        X - mu[k], np.dot(np.linalg.inv(s) , (X - mu[k]).T).T )
#a3 = np.exp(-.5 * np.einsum('ij, ij -> i',\
#        X - mu[k], np.dot(np.linalg.inv(s) , (X - mu[k]).T).T))
#
#b = (np.linalg.det(s) * ((2 * np.pi) ** (-X.shape[1]/2.)) ** (-.5))
## Vectorized implementation of e-step equation to calculate the 
## membership for each of k -gaussians
    for k in range(c):
        R[:, k] = w[k] * P(mu[k], Sigma[k])
    
    ### Likelihood computation
    log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
    
    log_likelihoods.append(log_likelihood)
    
    ## Normalize so that the responsibility matrix is row stochastic
    R = (R.T / np.sum(R, axis = 1)).T
    
    ## The number of datapoints belonging to each gaussian            
    N_ks = np.sum(R, axis = 0)
    
    k=1
    # M Step
    ## calculate the new mean and covariance for each gaussian by 
    ## utilizing the new responsibilities
    for k in range(c):
    
    ## means
        mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
        x_mu = np.matrix(X - mu[k])
        
        ## covariances
        Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
        
        ## and finally the probabilities
        w[k] = 1. / n * N_ks[k]
    # check for onvergence
#    if len(log_likelihoods) < 2 : continue
#    if np.abs(log_likelihood - log_likelihoods[-2]) <  eps: break
####################
### sklearn ########
####################
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, s=40)
plt.scatter(X[:,0], X[:,1], X[:, 2], c=labels, s=40)
plt.scatter(X[:,0], X[:,1], X[:, 2], c=labels)
n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]
          
n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
          
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
          
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')