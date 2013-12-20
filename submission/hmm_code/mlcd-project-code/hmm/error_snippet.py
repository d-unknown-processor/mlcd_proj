__author__ = 'arenduchintala'
import numpy as np

x = np.array([[4, 2, 0.6], [4.2, 2.1, 0.59], [3.9, 2.0, 0.58], [4.3, 2.1, 0.62], [4.1, 2.2, 0.63]])
print x
mu_x = np.mean(x,axis=0)
cov_x = np.cov(np.transpose(x))
print mu_x
print cov_x
generated_x = np.random.multivariate_normal(mu_x, cov_x)
print generated_x

