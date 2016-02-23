from numpy import *


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


def part2-forward(X):
    
def part2():
    '''
    N - number of samples
    b - 10 x 1
    x - (1 + 784) x N ; extra vector for bias
    output - 10 x 1

    for each possible outputs o_j, j = 0 -> 9: #output level
        for each training image M["trainj"][i], i = 1 -> len(M["trainj"]): #training image level
                 get image M["trainj"][i]
                 divide all pixels by 255.0
                 flatten image to create vector
                 o_j = sum(w)

    '''
