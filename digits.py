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


#import cPickle

import os
from scipy.io import loadmat




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
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def part2(W, B):
    '''
    N - number of samples
    b - 10 x 1
    x - (784) x N
    output - 10 x 1
    for each possible outputs o_j, j = 0 -> 9: #output level
        for each training image M["trainj"][i], i = 1 -> len(M["trainj"]): #training image level
                 get image M["trainj"][i]
                 divide all pixels by 255.0
                 flatten image to create vector
                 o_j = equation
    '''
    outputs = []
    # each value output[digit] is [trainimage1_output, trainimage2_output ... ,trainimagen_output]
    softmax_output = []    
    
    #generate list of weights - 10 weights per pixel
    # i.e. W[50][8] = weight of the 51st pixel to output digit 8

    for digit in range(10): # for each digit 
        train_key_name = "train" +  str(digit) # produce key for M
        
        for i in range(len(M[train_key_name])):
            #print(i)
            X = M[train_key_name][i].flatten() # input image array as vector
            
            X = X*(1.0/255.0) # scale the image array
            
            print(dot(W[0][digit], X))
            # adds output for ith training image to {i:[outputs]}
            outputs[digit].append(dot(W[0][digit], X) + B[0][digit]) # summation equation
            break
        # once all outputs for digit i are added to outputs[digit],
        # outputs[digit] should be [o1,o2, ... on], n = # images
        #print(outputs[digit][0])
        # make all cases of digit outputs into a matrix
        # but, transpose from M x N to N x M
        # (where N = # outputs for each single case, M = #cases)
        #softmax_input = matrix((outputs[digit])).T
        #softmax_output.append(softmax(softmax_input))
        
        
    #print(softmax_output)

def get_data():
    #Load the MNIST digit data
    M = loadmat("mnist_all.mat")

    train_data = []
    test_data = []

    train_target = []
    test_target = []

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
    
    print(train_data.shape)
    print(test_data.shape)
    print(train_target.shape)
    print(test_target.shape)
            
if __name__ == "__main__":
    get_data()

    #B = random.rand(1, 10)
    #W = random.rand(10, 784)

    #part2()
    