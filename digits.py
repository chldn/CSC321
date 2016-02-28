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

random.seed(100)

digit_to_numInputs = {} #key is the digit, value is the number of inputs for that digit
digit_to_indexPlusOne = [] #index represents the digit, value at index i is the index at which the inputs start in train_data, train_target etc..

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

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
    
def cost(y, t):
    '''
    t - target 
    y - output
    '''
    return -(1/60000.)*sum(t*log(y)) 

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
    train_data = array(train_data)*(1/255.0) # dimensions = (60000, 784)
    test_data = array(test_data)*(1/255.0) # dimensions = (10000, 784)
    train_target = array(train_target) # dimensions = (60000, 10)
    test_target = array(test_target) # dimensions = (10000, 10)
    
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
            #don't show axes and make image grayscale
            plt.gray()
            plt.axis('off')
    plt.savefig('100_dataset_images.jpg')

    
def generate(digit, Y, training):
    '''
    Returns the training data and train targets for a certain digit.
    
    digit - digit that you want to get the data for
    training - #bool representing if training / test data wanted

    Returns: (X, T, Y)
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

def get_output_part2(X, W, B):
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
    #print(X.shape)
    #print(T.shape)
    #print(Y.shape)
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
 

        
def get_dWs(Y, X, T):
    return derivative(Y.T, T.T, X.T)


def get_dWs_old(Y_total):
    '''
    Returns a list of 10 (10 x 784) matrices,
    each representing the Ws for each digit
    '''

    dWs = []
    total_dW = []

    for digit in range(0,10):

        #print(digit)
        X, T, Y = generate(digit, Y_total, training =True) 

        d = derivative(Y.T, T.T, X.T)
        dWs.append(d)
        if digit == 0:
            total_dW = d
        else:
            add(total_dW, d) 
    
    
    #print(total_dW.shape)
    #print(sum(total_dW))
    return asarray(total_dW)

        
def part5(train_data, train_target):
    '''
    minimize your the cost function using mini-batch gradient descent, using
    the training set provided to you
    learning rate = 0.01
    batch size = 50
   
    For the training and the test set, graph the negative-log probability of
    the correct answer and correct classification rate versus the number of
    updates to the weights and biases during training
   
    '''
    #create a randomized batch of size 50
    X, Y = create_batches(50, train_data, train_target)
    #start out with some randomized weights
    init_W = random.rand(784, 10)
    minimized = minibatch_grad_descent(X[0], Y[0], init_W)
    return minimized

 
def create_batches(k, train_data, train_target):
    '''
    Return a set of batches of size k, and a set of corresponding target values
    '''
    order = range(60000)
    random.shuffle(order) #randomly shuffle the order of the training examples
    set_batch_data = [] #set of all batches of size k of training examples
    set_batch_target = [] #corresponding set of all batches target values
    while len(order) >= k:
        #initialize corresponding empty lists for data and targets
        batch_data, batch_target = [], []
        #grab first k numbers from order, delete them from order
        batch_inds, order = order[:k], order[k:]
        for ind in batch_inds:
            batch_data.append(train_data[ind])
#             if len(batch_data)==2:
#                 print shape(batch_data)
#                 break
            batch_target.append(train_target[ind])
        set_batch_data.append(matrix(batch_data))
        set_batch_target.append(matrix(batch_target))
    return set_batch_data, set_batch_target
    
   
def minibatch_grad_descent(X, Y, init_W):
    '''
    Compute the gradient descent given a minibatch of the training data (X),
    the target values for that data (Y), and the initial weights (init_W)
    '''
    alpha = 0.01
    EPS = 1e-5   #EPS = 10**(-5)
    prev_W = init_W-10*EPS
    W = init_W.copy()
    while norm(W - prev_W) >  EPS:
        prev_W = W.copy()
        W -= alpha*derivative(Y, W, X)
   
    return W

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

def get_finite_diff(X, W, i, j, T):
    '''
    Returns dW using finite difference approximate of the gradient of the cost 
    with respect to W, at coordinate i
    '''
    
    h = zeros(W.shape)
    diff = 0.001

    h[i][j] = diff
    
    Y_plus = get_output_part2(X, W+h, B)
    Y_less = get_output_part2(X, W-h, B)
    
    #print(cost(Y_plus, T))
    #print(cost(Y_less, T))


    finite_diff = (cost(Y_plus, T) - cost(Y_less, T))/ 2*diff
    
    return finite_diff
    
def check_dWs(W, B, Y_total, dWs, T):
    # get dWs from gradient used to check with finite diff
    # outputs for rand_dn = (dW, j, i), where j is the digit, i is the index of input
    rand_d1 = next((dWs[0][i], 0, i) for i in range(784) if dWs[0][i] > 0)
    rand_d2 = next((dWs[3][i], 3, i) for i in range(784) if dWs[3][i] > 0)
    rand_d3 = next((dWs[6][i], 6, i) for i in range(784) if dWs[6][i] > 0)
    rand_d4 = next((dWs[7][i], 7, i) for i in range(784) if dWs[7][i] > 0)
    rand_d5 = next((dWs[9][i], 9, i) for i in range(784) if dWs[9][i] > 0)

    #print((rand_d1[2], rand_d2[2], rand_d3[2], rand_d4[2], rand_d5[2]))
    #print(dWs[0][13])
    gradient_dWs = (rand_d1[0], rand_d2[0], rand_d3[0], rand_d4[0], rand_d5[0])

    #print(rand_d1[2])

    fin_diff_d1 = get_finite_diff(X, W, rand_d1[2], 0, T)
    fin_diff_d2 = get_finite_diff(X, W, rand_d2[2], 3, T)
    fin_diff_d3 = get_finite_diff(X, W, rand_d3[2], 6, T)
    fin_diff_d4 = get_finite_diff(X, W, rand_d4[2], 7, T)
    fin_diff_d5 = get_finite_diff(X, W, rand_d5[2], 9, T)

    fin_diff_dWs = (fin_diff_d1, fin_diff_d2, fin_diff_d3, fin_diff_d4, fin_diff_d5)

    print("NW: ", gradient_dWs)
    print("FD: ", fin_diff_dWs)

def finite_diff(W, B, Y, T):
    fin_diffs = []
    #for digit in range(10):
    for i in range(784):
        
        fin_diff = get_finite_diff(X, W, i, 8, T)
        fin_diffs.append(fin_diff)
        print(i, fin_diff)

    fin_diffs = asarray(fin_diffs)
    print(fin_diffs)
    return fin_diffs
 
if __name__ == "__main__":
    #part1()
    train_data, test_data, train_target, test_target = get_data()
    W = (random.rand(784, 10)-0.5)*0.2
    B = (random.rand(1, 10)-0.5)*0.2

    Y = get_output_part2(train_data, W, B)
    
    X = train_data
    T = train_target
    dWs = get_dWs(Y, X, T) # get derivatives of all weights
    
    #plt.imshow(dWs[8].reshape(28,28))



    #plt.imshow(finite_diff(W, B, Y, T).reshape(28,28))
    #show()
    #check_dWs(W, B, Y, dWs, T)

    print(part5(train_data, train_target))

    #print(dWs.shape)
    
    
    

    

    #ft_dWs = get_finite_diff_dW()
    

    
    
