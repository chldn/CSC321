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


def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    output = softmax(L1)
    return L0, L1, output
    
def cost(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y.
    y is an NxM matrix where N(rows) is the number of outputs for a single case, 
    and M(col) is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
            

def get_data():
    '''
    This function puts all data from mnist_all.mat into the 2 lists train_data and test_data.
    In addition, the expected/target outputs are put into the two lists train_target and test_target. 
    The indices of (train_data, train_target) and (test_data, test_target) are the same so we can
    match the actual outputs later on to measure performance.
    '''
    #initiate lists for aggregate training and test arrays
    train_data = []
    test_data = []

    train_target = []
    test_target = []

    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    # getting 10 x 10 identity matrix to represent target outputs for networks
    identity_matrix = identity(10)

    for digit in range(0,10): 
        # training data input
        train_key = "train" + str(digit) # produce training key for M
        train_size = M[train_key].shape[0] # get number of cases / inputs
        train_data.extend(M[train_key]) # add to aggregated train data

        # target training data output; add in train_size number of identity[digit]s
        # for example, identity_matrix[3] = [0,0,0,1,0,0,0,0,0,0] <- expected / target output
        train_target.extend(list(tile(identity_matrix[digit], (train_size, 1))))

        # test data input 
        test_key = "test" + str(digit) # produce training key for M
        test_size = M[test_key].shape[0] # get number of cases / inputs
        test_data.extend(M[test_key]) # add to aggregated test data

        # target test data output; add in test_size number of identity[digit]s
        test_target.extend(list(tile(identity_matrix[digit], (test_size, 1))))

    # make all lists of aggregated arrays into a matrix & scale it 
    train_data = matrix(train_data)*(1/255.0) # dimensions = (60000, 784)
    test_data = matrix(test_data)*(1/255.0) # dimensions = (10000, 784)
    train_target = matrix(train_target)*(1/255.0) # dimensions = (60000, 10)
    test_target = matrix(test_target)*(1/255.0) # dimensions = (10000, 10)
    
    #print(train_data.shape)
    #print(test_data.shape)
    #print(train_target.shape)
    #print(test_target.shape)

    return train_data, test_data, train_target, test_target

def part1(data_set):
    '''
   Create a folder for each digit with ten images of that digit from the data set
   '''
#     for digit in range(10):
#         #d = os.path.dirname('image_folder' + str(digit))
#         if not os.path.exists('image_folder' + str(digit)):
#             os.makedirs('image_folder' + str(digit))
#             for i in range(10):
#                 #subplot(digit, i, digit+i), imshow(data_set[digit][i])
    pass
#                 #imsave('image_folder' + str(digit)+'/'+str(digit) + '.' +str(i)+'.jpg', data_set[digit][i])

def part2(X, W, B):
    '''
   This is the basis of a simple neural network.
   Return o's as linear combinations of the input X's (i.e. activation function is the identity)
 
   X - input of dimension 60000 x 784
   W - input of dimension 784 x 10
   B - input of dimension 1 x 10
   '''
    output = dot(X, W) + B # dimensions = (60000, 10) for train_data
    # print(softmax(output.T).shape)
    # print(softmax(output.T))
   
    return softmax(output.T)
 
def gradient_descent(W, derivative):
    pass

def derivative(W):
    pass
    
def part3(X, W, B):
    '''
   Return the gradient of the cross-­entropy cost function with respect to the
   parameters of the network (W and b), for a given subset of training cases.
   
   X - input of dimension 60000 x 784
   W - input of dimension 784 x 10
   B - input of dimension 1 x 10
   '''
    pass
    #cost = -sum(targets*log(predictions))
 
 
if __name__ == "__main__":
    train_data, test_data, train_target, test_target = get_data()
    #part1(train_data)
    W = random.rand(784, 10)
    B = random.rand(1, 10)
    part2(train_data, W, B)