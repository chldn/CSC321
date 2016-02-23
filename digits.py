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

#Load the MNIST digit data
M = loadmat("mnist_all.mat")


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


#def part2-forward(X):
    
def part2():
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
                 o_j = sum(w)
    '''
    outputs = {0:[], 1:[], 2:[], 3:[], 4:[], 
               5:[], 6:[], 7:[], 8:[], 9:[]} # dict of lists of outputs per digit i.e. outputs = [ [outputs for 1], [outputs for 2] ... ]
    average_output = []    
    
    #generate list of weights - 10 weights per pixel
    # i.e. W[50][8] = weight of the 51st pixel to output digit 8
    B = random.rand(1, 10)
    W = random.rand(10, 784)
    outputs[2]
    
    
    for digit in range(10): # for each digit 
        train_key_name = "train" + str(digit) # produce key for M
        print(digit)
        for i in range(len(M[train_key_name])):
            #print(i)
            X = M[train_key_name][i].flatten() # input image array as vector
            
            X = X*(1.0/255.0)
            #outputs[digit]
            #dot(W[0][digit], X)
            #B[digit]
            outputs[digit].append(dot(W[0][digit], X) + B[0][digit])
        average_output.append(average(outputs[digit]))
    print(average_output)
            
if __name__ == "__main__":
    part2()
    
        