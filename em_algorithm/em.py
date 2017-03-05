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
    # mu = [random.random() for i in range(k)]
    mu = [-0.1, 0.1]
    sd = [1 for i in range(k)]
    cp = [1.0/k] * k
    n = np.shape(x)[0]
    
    # i = class index
    # j = data point index
    for run in range(200):
        
        # probability of class i given jth data point
        p_ij = []
        
        # exceptation calculation
        for k_i in range(k):
            pdf = ss.norm(mu[k_i], sd[k_i]).pdf(x)
            p_ij.append(np.multiply(pdf, cp[k_i]))
        
        # maximization calculation
        p_tot = p_ij[0]
        for k_i in range(1,k):
            p_tot = np.add(p_tot, p_ij[k_i])
        
        p_ij = [p_ij[k_i]/p_tot for k_i in range(len(p_ij))]

        for k_i in range(k):
            sum_p_ij = np.sum(p_ij[k_i])
            mu[k_i] = np.sum(np.multiply(p_ij[k_i], x))/sum_p_ij
            sd[k_i] = np.sqrt(np.sum(np.multiply(p_ij[k_i], np.square(x - mu[k_i])))/sum_p_ij)
            cp[k_i] = sum_p_ij/n
        
        # TODO: Verficiation needed::
        # log-likelihood calculation
        for index in range(1, len(p_ij)):
            p_ij += p_ij[index]

        ll = np.sum(np.log(p_ij))
        print ll
        
    print cp
    print mu, sd
        