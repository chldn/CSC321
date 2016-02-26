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

digit_to_numInputs = {} #key is the digit, value is the number of inputs for that digit
digit_to_indexPlusOne = [] #index represents the digit, value at index i is the index at which the inputs start in train_data, train_target etc..


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

        # if digit is 0, just add the index of the train_size of 0
        # otherwise, add to the existing index number
        digit_to_indexPlusOne.append(train_size if digit == 0 else (digit_to_indexPlusOne[digit-1] + train_size))
        digit_to_numInputs[digit] = train_size

    # make all lists of aggregated arrays into a matrix & scale it 
    train_data = matrix(train_data)*(1/255.0) # dimensions = (60000, 784)
    test_data = matrix(test_data)*(1/255.0) # dimensions = (10000, 784)
    train_target = matrix(train_target)*(1/255.0) # dimensions = (60000, 10)
    test_target = matrix(test_target)*(1/255.0) # dimensions = (10000, 10)
    
    #print(digit_to_indexPlusOne)
    print(digit_to_numInputs)
    #print(train_data.shape)
    #print(test_data.shape)
    #print(train_target.shape)
    #print(test_target.shape)

    return train_data, test_data, train_target, test_target

def part1():
    '''
   Create a subplot image with ten images of each digit from the data set
   '''
    for digit in range(10):
        for i in range(10):
            image = M["train"+str(digit)][i]
            reshaped = reshape(image, [28, 28])
            subplot(10, 10, (digit)*10+(i+1))
            plt.imshow(reshaped)
        #don't show axes
    plt.axis('off')
    plt.savefig('100_dataset_images.jpg')

    
def generate(digit, Y, training):
    '''
    Returns the training data and train targets for a certain digit.
    
    digit - digit that you want to get the data for
    training - #bool representing if training / test data wanted
    '''
    if digit == 0:
        (start, end) = (0,digit_to_indexPlusOne[0])  
    else:
        (start, end) = (digit_to_indexPlusOne[digit-1], 
                            digit_to_indexPlusOne[digit])    
    if (training):
        return train_data[start:end], train_target[start:end], Y[start:end]
    else: # testing
        return test_data[start:end], test_target[start:end], Y[start:end]

def part2(X, W, B):
    '''
   This is the basis of a simple neural network.
   Return o's as linear combinations of the input X's (i.e. activation function is the identity)
 
   X - input of dimension 60000 x 784
   W - input of dimension 784 x 10
   B - input of dimension 1 x 10
   '''
    output = dot(X, W) + B # dimensions = (60000, 10) for train_data
    # print(softmax(output).shape)
    # print(softmax(output))
   
    return softmax(output)
 
def gradient_descent(W, derivative):
    pass

def derivative(Y, T, X):
    '''
    Return derivative(dW) of the cost function with respect to W.
    
    dW has dimensions 10 x 784
    '''
    return dot((Y - T), X.T)
    
def part3(X, W, B):
    '''
    Return the gradient of the crossentropy cost funciton with respect to the
    parameters of the network (W and b), for a given subset of training cases.
    X - input of dimension 60000 x 784
    W - input of dimension 784 x 10
    B - input of dimension 1 x 10
    '''
    pass
    #cost = -sum(targets*log(predictions))
 

        
def get_dWs(Y_total):
    '''
    Returns a list of 10 (10 x 784) matrices,
    each representing the Ws for each digit
    '''
    dWs = []
    
    for digit in range(0,10):
        X, T, Y = generate(digit, Y_total, training =True) 
        dWs.append(derivative(Y.T, T.T, X.T))

        #print(Y.shape) #(5958, 10) for digit = 2
        #print(X.shape) #(5958, 10)
        #print(T.shape) #(5958, 784)
    
    # debugging
    i = 0
    for dW in dWs: # should output 10 (10, 784)s (one for each digit)
        print(i, dW.shape)
        i += 1
    #end debug
    
    return dWs
    
def part4(X, W, B):
    '''
    Evaluate gradient using part3 function with finite differences, and make 
    sure you get the same result either way
    '''
    if part3(X, W, B) == finite_diff(X, W, B):
        return True
        
def finite_diff():
    '''
    approximate the gradient of the cost with respect to W, at coordinate i
    '''
    cost = cost(part2(X, W, B), Y)

    pass
        
def part5():
    '''
    minimize your the cost function using mini-batch gradient descent, using 
    the training set provided to you
    learning rate = 0.01
    batch size = 50
    
    For the training and the test set, graph the negative-log probability of 
    the correct answer and correct classification rate versus the number of 
    updates to the weights and biases during training
    
    for 20 examples from the test set, add them 
    '''
    pass

def part7():
    '''
    For a network which takes X (input vector of 784 units), H (hidden
    layer of 300 hidden units), O (the 10 output units - one per digit, W1 (the 
    weights from X to H),and W2 (the weight matrix going from H to O)..
    Implement a function that computes the gradient of the negative log 
    probability of the correct answer function, computed for a set of training 
    examples, with respect to the network parameters
    '''
    
    pass
    
 
if __name__ == "__main__":
    train_data, test_data, train_target, test_target = get_data()
    #part1(train_data)
    W = random.rand(784, 10)
    B = random.rand(1, 10)
    Y_total = part2(train_data, W, B)
    #print("Ex. softmax of first input = \n" + str(Y[:1]))
    
    dWs = get_dWs(Y_total) # get derivatives of all weights
    

    
    
