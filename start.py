import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
from itertools import product



# *-------* Kernel functions for data preprocessing

alphabet = 'ACGT'
k = 3 # |subsets| in spectrum

## get spectrum of string
def getSpectrum(sequence, k):
    # length of spectrum substracts (k-1) elements to length
    spec = [0] * (len(sequence) - (k-1))   
    for i in range(len(spec)): # part string into k-tuples 
        spec[i] = sequence[i:i+k]
    return spec

## create all possible combinations (cartesian product)
def cartProd(alphabet, k):
    prod = product('ACGT',repeat=k)
    comb = []
    for p in prod:
        # turn into string
        comb.append(''.join(p))
    return comb

## replacing sequence by occurrence of k-spectra
def occurrances(sequence, k):
    spec = getSpectrum(sequence, k)
    comb = cartProd(alphabet, k)
    occ = [0] * (len(comb)) # = (len(alphabet)^k)
    for i,s in enumerate(comb):
        occ[i] = spec.count(comb[i])
    return occ


def kernel_method(data, k):
    # get spectrum for each sequence
    spec = [0] * len(data)
    for i, seq in enumerate(data['seq']):
        spec[i] = getSpectrum(seq, k)
        
    # all combinations of alphabet of length k
    comb = cartProd(alphabet, k)
    
    # convert spectrum into occurrances
    phi = [0] * len(spec)
    for j, row in enumerate(spec):
        occ = [0] * len(comb)
        for i, string in enumerate(comb):
            occ[i] = row.count(string)
        phi[j] = occ
    
    # replace sequence by occurrance array
    data['seq'] = phi
    
    return data



# *-------* Activation and loss functions (and their derivatives)

def sigmoid(x):
    # Sigmoid activation function 
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    # Derivative of sigmoid 
    f = sigmoid(x)
    return f * (1 - f)

def tanh(x):
    # Hyperbolic tangent activation function
    return 2 * sigmoid(2*x) - 1

def d_tanh(x):
    # Derivative of tanh
    return 1 - tanh(x)**2

def bin_ce_loss(y, y_hat):
    # Binary cross-entropy loss
    if y == 1:
        return -np.log(y_hat)
    else:
        return -np.log(1 - y_hat)

def d_bin_ce_loss(y, y_hat):
    # Derivative of binary cross-entropy loss
    if y == 1:
        return y_hat - 1
    else:
        return y_hat

def bin_ce_loss_multidim(y, y_hat):
    # Binary cross-entropy loss for multidimentional inputs
    # returns the mean of all losses
    loss = 0
    for yTrue, yHat in zip(y, y_hat):
        loss += bin_ce_loss(yTrue, yHat)
        
    return loss / y.shape[0]

def mse(y, y_hat):
    # Mean squared error loss function
    return ((y - y_hat)**2).mean()



# *-------* Neural network

class NN:
    '''
    Neural network with :
    - 64 input nodes
    - 1 hidden layer with 44 nodes
    - an output layer with 1 node
    
    - W1 is an array of weights for hidden layer
    - W2 is an array of weights for output layer
    - B is a bias array of 45 elements, where the last element is a bias for an output node
    - X is a numpy array of inputs with 64 elements.
    - sum_H is a dot product between W and X plus bias
    - H is hidden layer with 44 nodes
    - y_hat is a predicted value
    
    - data is a (n x 64) numpy array of observations
    - all_y_true is a numpy array with n labels
    - all_y_hat  is a numpy array with n predictions
    '''
    def __init__(self):
        
        ### *---* Initialize weights using Xavier normal initialization
        
        nb_input_nodes, nb_hidden_nodes, nb_output_nodes = 64, 44, 1
        
        # Weights for hidden layer
        stddev_w1 = np.sqrt(2 / (nb_input_nodes + nb_hidden_nodes))
        lower_w1 = -stddev_w1
        upper_w1 =  stddev_w1
        # truncated normal
        tn_w1 = truncnorm(lower_w1 / stddev_w1, upper_w1 / stddev_w1, loc=0, scale=stddev_w1) 
        
        # Weights for output layer
        stddev_w2 = np.sqrt(2 / (nb_hidden_nodes + nb_output_nodes))
        lower_w2 = -stddev_w2
        upper_w2 =  stddev_w2
        # truncated normal
        tn_w2 = truncnorm(lower_w2 / stddev_w2, upper_w2 / stddev_w2, loc=0, scale=stddev_w2) 
        
        self.W1 = tn_w1.rvs(nb_input_nodes  * nb_hidden_nodes)
        self.W2 = tn_w2.rvs(nb_hidden_nodes * nb_output_nodes)
        
        ### *---* Initialize biases
        self.B = np.zeros((nb_hidden_nodes + nb_output_nodes, ))

    def feedforward(self, X):
        
        nb_input_nodes, nb_hidden_nodes, nb_output_nodes = 64, 44, 1
        
        # < W,X > + b
        self.sum_H = np.zeros((nb_hidden_nodes,))
        for i in range(nb_hidden_nodes):
            self.sum_H[i] = np.dot(self.W1[nb_input_nodes*i : nb_input_nodes*(i+1)], X) + self.B[i]

        # hidden layer nodes
        self.H = tanh(self.sum_H) 
        
        # < W, H > + b
        self.sum_yhat = np.dot(self.W2, self.H) + self.B[-1]
        
        # output
        self.y_hat = sigmoid(self.sum_yhat)
        return self.y_hat
    
    def train(self, data, all_y_true):
        
        learn_rate = 0.01
        iterations = 2000
        
        nb_input_nodes, nb_hidden_nodes, nb_output_nodes = 64, 44, 1
        

        for iteration in range(iterations):
            for x, y in zip(data, all_y_true):
                
                self.feedforward(x)
                
                ### *---* Calculate partial derivatives
                
                # partial Loss / partial y_hat
                d_L_d_yhat = d_bin_ce_loss(y, self.y_hat)

                # partial y_hat / partial sum_yhat
                d_yhat_d_sum_yhat = d_sigmoid(self.sum_yhat)
                
                # partial Loss / partial W1
                d_L_d_W1 = d_L_d_yhat * d_yhat_d_sum_yhat * np.ones((nb_hidden_nodes,nb_input_nodes))
                
                for i in range(nb_hidden_nodes):
                    for j in range(nb_input_nodes):
                        d_L_d_W1[i,j] *= self.W2[i] * d_tanh(self.sum_H[i]) * x[j]
                
                d_L_d_W1 = d_L_d_W1.reshape(nb_hidden_nodes * nb_input_nodes) # = 44*64
                
                # partial Loss / partial W2
                d_L_d_W2 = d_L_d_yhat * d_yhat_d_sum_yhat * self.H
                
                # partial Loss / partial B1
                d_L_d_B1 = d_L_d_yhat * d_yhat_d_sum_yhat * self.W2 * d_tanh(self.sum_H) * np.ones((nb_hidden_nodes, ))
                
                # partial Loss / partial B2
                d_L_d_B2 = d_L_d_yhat * d_yhat_d_sum_yhat * 1
                
                ### *---* Update weights and biases
                self.W1 -= learn_rate * d_L_d_W1
                self.W2 -= learn_rate * d_L_d_W2
                
                self.B[:nb_hidden_nodes] -= learn_rate * d_L_d_B1
                self.B[nb_hidden_nodes]  -= learn_rate * d_L_d_B2
                
            # Calculate total loss at the end of each iteration
            all_y_hat = np.apply_along_axis(self.feedforward, 1, data)
            
            mseloss =                  mse(all_y_true, all_y_hat) # MSE loss
            bceloss = bin_ce_loss_multidim(all_y_true, all_y_hat) # binary CE loss
            skacc =         accuracy_score(all_y_true, np.round(all_y_hat)) # accuracy score
            
            # Show the score on test data at the end of each iteration
            print("%d LOSS: mse %.3f bce %.3f ACC: %.3f" % (iteration, mseloss, bceloss, skacc))
            


# *-------* Main program

# prepare test data

x_te0 = pd.read_csv('data/Xte0.csv')
x_te1 = pd.read_csv('data/Xte1.csv')
x_te2 = pd.read_csv('data/Xte2.csv')

x_te = x_te0.append(x_te1, ignore_index=True)
x_te = x_te.append(x_te2, ignore_index=True)

x_te_num = kernel_method(x_te, 3)
x_te_num = x_te_num.seq

x_test = np.asarray(x_te_num.tolist())



# Set pretrained weights and biases
nn = NN()
nn.W1 = genfromtxt('W1.csv', delimiter=',')
nn.W2 = genfromtxt('W2.csv', delimiter=',')
nn.B =  genfromtxt('B.csv',  delimiter=',')


# Save predictions to Yte.csv
all_y_hat = np.apply_along_axis(nn.feedforward, 1, x_test)
f = open('Yte.csv','w')
f.write('Id,Bound\n')

for i in range(all_y_hat.shape[0]):
    f.write('%s,%s\n' % (i, 
                         0 if all_y_hat[i] < 0.5 else 1))
f.close()


