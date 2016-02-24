from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import cPickle

import os
from scipy.io import loadmat



def get_data():
    train_data = []
    test_data = []

    train_target = []
    test_target = []

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    identity_matrix = identity(10)

    for digit in range(0,10): 
        # training data input
        train_key = "train" + str(digit) # produce training key for M
        train_size = M[train_key].shape[0] # get number of cases / inputs
        train_data.extend(M[train_key]) # add to aggregated train data

        # target training data output; add in train_size number of identity[digit]s
        train_target.extend(list(tile(identity_matrix[digit], (train_size, 1))))

        # test data input 
        test_key = "test" + str(digit) # produce training key for M
        test_size = M[test_key].shape[0] # get number of cases / inputs
        test_data.extend(M[test_key]) # add to aggregated test data

        # target test data output; add in test_size number of identity[digit]s
        test_target.extend(list(tile(identity_matrix[digit], (test_size, 1))))


    train_data = matrix(train_data)*(1/255.0) # dimensions = (60000, 784)
    test_data = matrix(test_data)*(1/255.0) # dimensions = (10000, 784)
    train_target = matrix(train_target)*(1/255.0) # dimensions = (60000, 10)
    test_target = matrix(test_target)*(1/255.0) # dimensions = (10000, 10)
    
    #print(train_data.shape)
    #print(test_data.shape)
    #print(train_target.shape)
    #print(test_target.shape)

    return train_data, test_data, train_target, test_target


# activation function
def sigmoid(t):
    return 1/(1+exp(-t))

def forward(x, W1, b1, W2, b2):
    print("dot(W1.T, x)", dot(W1.T, x))
    h = sigmoid(b1+dot(W1.T, x))    
    print("h=", h)
    out = sigmoid(b2+dot(W2.T, h))
    print("out =", out)
    return out

    
def cost(out, y):
    return (out-y)**2


def tutorial() :
    x = array([1, -2]) #input

    W1 = array([[1,  -3],
                [-2, -2]])
    b1 = array([-7, .5])        

    W2 = array([[-2], [5]])
    b2 = array([[-4]])

    dx = array([0, 0.001])

    #
    dW2 = array([[0], [0.001]])
    dW2 = array([[0.001], [0]])

    dW1 = array([[0, 0],
                 [0, 0.001]])
    #

    #Estimate the derivative w.r.t x2
    print(  (cost(forward (x+dx, W1, b1, W2, b2),1)-
             cost(forward (x, W1, b1, W2, b2),1))/0.001)


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y.
    y is an NxM matrix where N(rows) is the number of outputs for a single case, 
    and M(col) is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
            

def part2(X, W, B):
    '''
    This is the basis of a simple neural network.
    Return o's as linear combinations of the input X's (i.e. activation function is the identity)

    X - input of dimension 60000 x 784
    W - input of dimension 784 x 10
    B - input of dimension 1 x 10
    '''
    output = dot(X, W) + B # dimensions = (60000, 10) for train_data
    #print(softmax(output.T).shape)
    return softmax(output.T)


if __name__ == "__main__":
    train_data, test_data, train_target, test_target = get_data()

    W = random.rand(784, 10)
    B = random.rand(1, 10)
    part2(train_data, W, B)


    