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


def get_data():
    '''
    This function puts all data from mnist_all.mat into the 2 lists train_data and test_data.
    In addition, the expected/target outputs are put into the two lists train_target and test_target. 
    The indices of (train_data, train_target) and (test_data, test_target) are the same so we can
    match the actual outputs later on to measure performance.

    @return:         Dimensions:
    train_data :     60000 x 784
    test_data :     0000 x 784

    train_target :     60000 x 10
    test_target :     10000 x 10
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


def cost(y, t):
    '''
    t - target 
    y - output
    '''
    #return (1./60000)*(-sum(t*log(y)))
    return -sum(t*log(y)) 


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y.
    y is an NxM matrix where N(rows) is the number of outputs for a single case, 
    and M(col) is the number of cases

    @return:    Dimensions:
    softmax:     1 x 10'''
    
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

# PART 2 - implement single layer network
def single_layer_network(X, W, B):
    '''
   This is the basis of a simple neural network.
   Return o's as linear combinations of the input X's (i.e. activation function is the identity)
 
   X - input of dimension 60000 x 784
   W - input of dimension 784 x 10
   B - input of dimension 1 x 10
   '''
    #print("p2 X", X)
    #print("p2 W", W)
    #print("p2 B", B)
    output = dot(X, W) + B # dimensions = (60000, 10) for train_data
    # print(softmax(output).shape)
    # print(softmax(output))
   
    return softmax(output)


# PART 3 - CROSS ENTROPY COST FUNCTION
def derivative(Y, T, X):
    '''    
    Return the gradient of the crossentropy cost function with respect to the
    parameters of the network (W and b), for a given subset of training cases.
    X - input of dimension 60000 x 784
    W - input of dimension 784 x 10
    B - input of dimension 1 x 10
    

    @Returns: 
    dW has dimensions 10 x 784
    '''
    return dot((Y - T), X.T)*10


def get_dWs(Y, X, T):
    return derivative(Y.T, T.T, X.T)


def derivative_b(Y, T, size):
    return dot((Y-T).T, ones((size,1)) )


# PART 4 - FINITE DIFFERENTIAL CHECKING
def part4(W, B, Y, dWs, T):
    '''
    Verifies tat Part 3 runs correctly by computing with derivative functino and with finite-difference
    approximation for 5 coordinates of W and b.
    '''
    check_dWs(W, B, Y, dWs, T)


def get_finite_diff(X, W, i, j, T):
    '''
    Returns dW using finite difference approximate of the gradient of the cost 
    with respect to W, at coordinate i
    '''
    h = zeros(W.shape)
    diff = 0.0001

    h[i][j] = diff
    
    Y_plus = single_layer_network(X, W+h, B)
    Y_less = single_layer_network(X, W-h, B)
    
    #print(cost(Y_plus, T))
    #print(cost(Y_less, T))

    #finite_diff = ((1./60000)*cost(Y_plus, T) - (1./60000)*cost(Y_less, T))/ diff**2
    finite_diff = ((1./60000)*cost(Y_plus, T) - (1./60000)*cost(Y_less, T))/ (diff*2)
    
    return finite_diff

    
def check_dWs(W, B, Y_total, dWs, T):
    '''
    This function is used to check the neural network gradient against the gradient found 
    through calculating the finite differentials.

    '''
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

    fin_diff_d1 = get_finite_diff(X, W, rand_d1[2], 0, T)
    fin_diff_d2 = get_finite_diff(X, W, rand_d2[2], 3, T)
    fin_diff_d3 = get_finite_diff(X, W, rand_d3[2], 6, T)
    fin_diff_d4 = get_finite_diff(X, W, rand_d4[2], 7, T)
    fin_diff_d5 = get_finite_diff(X, W, rand_d5[2], 9, T)

    fin_diff_dWs = (fin_diff_d1, fin_diff_d2, fin_diff_d3, fin_diff_d4, fin_diff_d5)

    print("NW: ", gradient_dWs)
    print("FD: ", fin_diff_dWs)


def finite_diff(W, B, Y, T):
    '''
    TEMPORARY FUNCTION USED FOR CHECKING
    Calculates all dfs for number 8
    '''
    fin_diffs = []
    #for digit in range(10):
    for i in range(784):
        
        fin_diff = get_finite_diff(X, W, i, 8, T)
        fin_diffs.append(fin_diff)
        print(i, fin_diff)

    fin_diffs = asarray(fin_diffs)
    print(fin_diffs)
    return fin_diffs


# PART 5 - MINI GRADIENT DIFFERENTIALS
def create_batches(k, data, target, size):
    '''
    Return a set of batches of size k, and a set of corresponding target values

    k - batch size
    '''
    order = range(size)
    random.shuffle(order) #randomly shuffle the order of the training examples
    set_batch_data = [] #set of all batches of size k of training examples
    set_batch_target = [] #corresponding set of all batches target values
    while len(order) >= k:
        #initialize corresponding empty lists for data and targets
        batch_data, batch_target = [], []
        #grab first k numbers from order, delete them from order
        batch_inds, order = order[:k], order[k:]
        for ind in batch_inds:
            batch_data.append(data[ind])
#             if len(batch_data)==2:
#                 print shape(batch_data)
#                 break
            batch_target.append(target[ind])
        set_batch_data.append(batch_data)
        set_batch_target.append(batch_target)
    return asarray(set_batch_data), array(set_batch_target)


def minibatch_grad_descent(train_data, train_target, init_W, init_B):
    '''
    Calculates gradient descent with respect to W and B in mini batches of 50.
    This runs through the 60000 images in of 1200 iterations of batch size 50.

    @returns:
    W : 10 x 784
    B : 10 x 1
    '''
    alpha = 0.01
    EPS = 1e-5
    
    W = init_W.copy()
    B = init_B.copy()
    b = 50


    X, T = create_batches(b, train_data, train_target, 60000)
    print(X.shape)
    print(T.shape)
    X_prev = []
    for i in range(1200): # for each batch of 50 images
        print(i)
        
        Y = single_layer_network(X[i], W, B) # calculate output based on the batch X[i] on Weights and Bias

        dW = derivative(Y.T, T[i].T, X[i].T) # find Cost derivative w respect to Wij
        #print((ones((10,784)) * (alpha*(1./b)*dW)).T)
        W = W - (ones((10,784)) * (alpha*(1./b)*dW)).T

        dB = derivative_b(Y, T[i], b).T
        B = B - (1./b)*alpha*dB
        #W = W - dW.T
        #B = B - dB

        if (i%20 == 0):
            imshow(W.T[5].reshape((28, 28)))
            show()

        # cost function increases
        if (i%20 == 0): 
            print(i, cost(Y,T), dW)
        
        prev_W = W.copy()
    return W, B


def check_performance(train_data, train_target, test_data, test_target, init_W, init_B):
    '''
    Goes through test set it batch size = 1000.
    Calculates the correctness of each test case, and outputs the total performance.

    @returns:
    total_peformance - percentage of correctness
    '''
    W, B = part5(train_data, train_target, init_W, init_B)
    b = 1000
    X_set, T_set = create_batches(b, test_data, test_target, 10000)
    #print(X_set[0].shape)
    #print(T_set[0].shape)

    Y = dot(X_set[0], W) + B
    

    for row in range(b):
        not_max = Y[row] < amax(Y[row]) # get all non_prediction indices
        Y[row][not_max] = 0 # set all non_predictions to 0

        the_max = Y[row] == amax(Y[row]) # get all non_prediction indices
        Y[row][the_max] = 1 # set all non_predictions to 1


    prediction_correctness = T_set[0] == Y

    correct = 0
    for input in range(b):
        if all(prediction_correctness[input]):
            correct+=1

    total_performance = float(correct)/b
    print(total_performance)

    return total_performance

def part5(train_data, train_target, init_W, init_B):
    '''
    minimize your the cost function using mini-batch gradient descent, using
    the training set provided to you
    learning rate = 0.01
    batch size = 50
   
    For the training and the test set, graph the negative-log probability of
    the correct answer and correct classification rate versus the number of
    updates to the weights and biases during training
   
    '''
    minimized = minibatch_grad_descent(train_data, train_target, init_W, init_B)
    return minimized


if __name__ == "__main__":
    #part1()

    train_data, test_data, train_target, test_target = get_data() 
    
    W = (random.rand(784, 10)-0.5)*0.02
    B = (random.rand(1, 10)-0.5)*0.02
    X = train_data
    T = train_target

    # PART 2
    Y = single_layer_network(X, W, B)

    # PART 3
    dWs = get_dWs(Y, X, T) # get derivatives of all weights
    
    #plt.imshow(dWs[8].reshape(28,28))
    #plt.imshow(finite_diff(W, B, Y, T).reshape(28,28))
    #show()
    
    # PART 4
    part4(W, B, Y, dWs, T)

    # PART 5
    #print(part5(train_data, train_target, W, B))
    check_performance(train_data, train_target, test_data, test_target, W, B)
    #print(dWs.shape)
    

    #ft_dWs = get_finite_diff_dW()