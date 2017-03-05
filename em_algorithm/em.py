import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


def plot(x, bins=40):
    plt.hist(x, bins=bins, histtype='bar')
    plt.show()

# data generation
np.random.seed(123)
data1 = np.random.normal(-10, 1, 4000)
data2 = np.random.normal(10, 1, 4000)
x = np.concatenate([data1, data2])
# plot(x)



for k in range(2, 3):
    
    # expectation initialization
    # k = number of centers
    # cp = class probability
    random.seed(12345)
    mu = [random.random() for i in range(k)]
    mu = [-1, 1]
    sd = [1 for i in range(k)]
    cp = [1.0/k for i in range(k)]
    
    # i = class index
    # j = data point index
    for run in range(200):
        
        # probability of class i given jth data point
        p_ij = []
        p_tot = []
        
        for k_i in range(k):
            pdf = ss.norm(mu[k_i], sd[k_i]).pdf(x)
            p_ij.append(np.multiply(pdf, cp[i]))
            p_tot.append(np.sum(p_ij[k_i]))
            mu[k_i] = np.sum(np.multiply(p_ij[k_i], x))/p_tot[k_i]
            sd[k_i] = np.sqrt(np.sum(np.multiply(p_ij[k_i], np.square(x - mu[k_i])))/p_tot[k_i])
            
        cp = p_tot/sum(p_tot)
        
        for index in range(1, len(p_ij)):
            p_ij += p_ij[index]
        
        # log-likelihood
        ll = np.sum(np.log(p_ij))
        print ll
        
    print cp
    print mu, sd
        