from __future__ import division
import numpy as np
import utilities as utils
import random


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def initZeroW(self, weights, bias):
        wzeros = [np.zeros(w.shape) for w in weights]
        bzeros = [np.zeros(b.shape) for b in bias]
        return wzeros, bzeros

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        '''
        :param parameters: dictionary of various parameters
        '''
        self.params = {
            'hidden_nw_str': [4],
            'mbs': 1,
            'stepsize': 0.01,
            'transfer': 'sigmoid',
            'cost':'crossentropyloss',
            'epochs': 1,
            'regularization': None,
            'regwt' : 0.1
        }
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)

        # activation function assingment
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        elif self.params['transfer'] is 'linear':
            self.transfer = utils.linear
            self.dtransfer = utils.dlinear
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> cannot handle your transfer function')

        # cost function assignment
        if self.params['cost'] == 'squareloss':
            self.dcost = utils.dsqloss
        elif self.params['cost'] == 'crossentropyloss':
            self.dcost = utils.dceloss
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> cannot handle loss function')

        # Regularization function assignment
        if self.params['regularization'] == None:
            self.regularizer = utils.noreg
        elif self.params['regularization'] == 'L1':
            self.regularizer = utils.regl1
        elif self.params['regularization'] == 'L2':
            self.regularizer = utils.regl2
        else:
            # For now, only allowing L1 and L2 regularization
            raise Exception('NeuralNet -> cannot handle regularlization')

        self.w, self.b = [], []

    def _initRandW(self,nw_str,):
        w = [np.random.randn(y, x) for x,y in zip(nw_str[:-1],nw_str[1:])]
        b = [np.random.randn(y, 1) for y in nw_str[1:]]
        return w, b

    def learn(self, Xtrain, ytrain,):

        ni = Xtrain.shape[1]
        no = ytrain.shape[1]
        hidden_nw_str = self.params['hidden_nw_str']
        nw_str = [ni] + hidden_nw_str + [no]
        self.nts = len(Xtrain)              # number of training samples
        self.w, self.b = self._initRandW(nw_str)
        self.lw = len(self.w)
        self.SGD(Xtrain,ytrain)

    def SGD(self, Xtrain, ytrain,):
        epochs = self.params['epochs']      # number of passes on train data
        eeta = self.params['stepsize']      # step size
        mbs = self.params['mbs']            # mini batch size
        nts = self.nts                      # no of training samples
        regwt = self.params['regwt']        # regularization parameter
        Z = [i for i in range(nts)]


        for epoch in range(epochs):
            # print(epoch)
            random.shuffle(Z)
            Xtrain, ytrain = Xtrain[Z], ytrain[Z]

            # generating mini batches
            mini_batches = [Z[k:k+mbs] for k in range(0, nts, mbs)]
            # mini_batches = [training_data[k:k+mbs] for k in range(0, nts, mbs)]

            for mini_batch in mini_batches:
                mini_batch_data = zip(Xtrain[mini_batch], ytrain[mini_batch])
                w_batch_update, b_batch_update = self.mini_batch_update(mini_batch_data)

                # update the weights: Note the weights are being regularized where as the biases are not

                # print self.w, self.b
                # print w_batch_update

                self.w = [w - ((eeta/mbs)*w_change) + self.regularizer(w, regwt, nts, eeta)
                        for w, w_change in zip(self.w, w_batch_update)]
                self.b = [b - ((eeta/mbs)*b_change)
                        for b, b_change in zip(self.b, b_batch_update)]

    def mini_batch_update(self, mini_batch):

        w_batch_update, b_batch_update = self.initZeroW(self.w, self.b)

        for x, y in mini_batch:

            w_sample_update, b_sample_update = self.backprop(x, y)

            for w in range(self.lw):
                # print(w_batch_update[w].shape, b_batch_update[w].shape)
                # print(w_sample_update[w].shape, b_sample_update[w].shape)
                w_batch_update[w] += w_sample_update[w]
                # print(b_batch_update[w].shape, b_sample_update[w].shape)
                b_batch_update[w] += b_sample_update[w]

        return w_batch_update, b_batch_update

    def backprop(self, x, y):

        x = np.atleast_2d(x).T
        y = np.atleast_2d(y).T

        w_sample_update, b_sample_update = self.initZeroW(self.w, self.b)

        lli, llo = self.feedforword(x)

        # Backward pass Last Layer
        delta = self.dcost(llo[-1], y) * self.dtransfer(lli[-1])
        w_sample_update[-1] = np.dot(delta, llo[-2].T)
        b_sample_update[-1] = delta

        # Backward pass Other Layers

        for w in range(2, self.lw+1):
            delta = np.dot(self.w[-w + 1].T, delta) * self.dtransfer(lli[-w])
            w_sample_update[-w] = np.dot(delta, llo[-w - 1].T)
            b_sample_update[-w] = delta

        return (w_sample_update, b_sample_update)

    def feedforword(self, ip):
        '''
        function for calculating the weights across various layers
        :param inputs:
        :return:
        '''
        lli=[]  # layer input list (lli: list of layer inputs)
        llo=[]  # layer ouput list (llo: list of layer outputs)

        li = ip
        llo.append(li)

        for w,b in zip(self.w, self.b):
            # print(w.shape, llo[-1].shape, b.shape)
            lli.append(np.dot(w, llo[-1]) + b)
            llo.append(self.transfer(lli[-1]))

        return lli, llo

    def predict(self, Xtest):
        predictions = []
        for x in Xtest:
            xT = np.atleast_2d(x).T
            # print self.feedforword(xT)[-1][-1]
            P = self.feedforword(xT)[-1][-1]
            print P[0][0] >= 0.5
            if P[0][0] >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        predictions = np.array(predictions)
        return predictions


# ---------------------------------------------------------------------------------------------

# -----------------------------------DATA ENTRY------------------------------------------------

# ---------------------------------------------------------------------------------------------

x = [[0,0,0,0],
     [0,0,0,1],
     [0,0,1,0],
     [0,0,1,1],
     [0,1,0,0],
     [0,1,0,1],
     [0,1,1,0],
     [0,1,1,1],
     [1,0,0,0],
     [1,0,0,1],
     [1,0,1,0],
     [1,0,1,1],
     [1,1,0,0],
     [1,1,0,1],
     [1,1,1,0],
     [1,1,1,1]]

y = [[0],[1],[1],[0],[1],[0],[0],[1],[1],[0],[0],[1],[0],[1],[1],[0]]
# y= [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]

# -------------------------------------------------------------------------------------------------------
#                         Test Case
# -------------------------------------------------------------------------------------------------------

# x = [[0,0,0],
#      [0,0,1],
#      [0,1,0],
#      [0,1,1],
#      [1,0,0],
#      [1,0,1],
#      [1,1,0],
#      [1,1,1]]
#
# y = [[0,0,1],
#      [0,1,0],
#      [0,1,1],
#      [1,0,0],
#      [1,0,1],
#      [1,1,0],
#      [1,1,1],
#      [0,0,0]]
# ------------------------------------------------------------------------------------------------------

# x = [[0,0],
#      [0,1],
#      [1,0],
#      [1,1]]
# y = [0,1,1,0]
# # y = [[0],[1],[1],[0]]

# ---------------------------------------------------------------------------------------------
x = np.array(x)
y = np.array(y)

# ----------------------------------------------------------------------------------------------
#                                 My Code
# -----------------------------------------------------------------------------------------------
print "x = ",x, "\n y = ",y
print x.shape, y.shape

params = {
    'hidden_nw_str': [4],
    'mbs': 1,
    'stepsize': 0.001,
    'transfer': 'sigmoid',
    'cost':'crossentropyloss',
    'epochs': 100000,
    'regularization': 'L2',
    'regwt' : 0.001
}

NN = NeuralNet(parameters=params)
NN.learn(Xtrain=x, ytrain=y)
ytest = NN.predict(Xtest=x)
y= [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]
print np.array(y)
print ytest

# ---------------------------------------------------------------------------------------------
#                             sklearn implementation
# ---------------------------------------------------------------------------------------------

# from sklearn import neural_network
#
# nn = neural_network.MLPClassifier(hidden_layer_sizes=(1,), activation='logistic')
# nn.fit(x,y)
# print nn.predict(x)