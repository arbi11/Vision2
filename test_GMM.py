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
    
c = 4 ## number of clusters
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
    
plt.plot(log_likelihoods)
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
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]
          
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')

####################
### Edward #########
####################

import edward as ed
import matplotlib.cm as cm
import six
import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture)

plt.style.use('ggplot')

def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[1, 1], [-1, -1]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


# N = 500  # number of data points
K = 4  # number of components
# D = 3  # dimensionality of data
#ed.set_seed(42)
D = d
N = n 
# x_train = build_toy_dataset(N)
x_train = X
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.axis([-3, 3, -3, 3])
plt.title("Simulated dataset")
plt.show()

pi = Dirichlet(tf.ones(K))
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat

T = 500  # number of MCMC samples
qpi = Empirical(tf.get_variable(
    "qpi/params", [T, K],
    initializer=tf.constant_initializer(1.0 / K)))
qmu = Empirical(tf.get_variable(
    "qmu/params", [T, K, D],
    initializer=tf.zeros_initializer()))
qsigmasq = Empirical(tf.get_variable(
    "qsigmasq/params", [T, K, D],
    initializer=tf.ones_initializer()))
qz = Empirical(tf.get_variable(
    "qz/params", [T, N],
    initializer=tf.zeros_initializer(),
    dtype=tf.int32))

inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},
                     data={x: x_train})
inference.initialize()

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
  t = info_dict['t']
  if t % inference.n_print == 0:
    print("\nInferred cluster means:")
    print(sess.run(running_cluster_means, {t_ph: t - 1}))
    

# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
sigmasq_sample = qsigmasq.sample(100)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

clusters = tf.argmax(log_liks, 1).eval()

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()



