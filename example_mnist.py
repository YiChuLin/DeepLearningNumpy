"""
This sample program uses the deep learning library to train on the MNIST Dataset
Author: Arvin Lin
"""
import mnist
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from lib import *

def one_hot(y):
	"""
	performs onehot conversion on an array
	ex: >>>y = np.array([1,2,3])
	    >>>y_h = one_hot(y)
	    >>>y_h
	    np.array([[0,1,0,0],
	    		  [0,0,1,0],
	    		  [0,0,0,1]]) 
	Usage: y_onehot = one_hot(y)
	"""
	yh = np.zeros((y.size, y.max()+1))
	yh[np.arange(y.size), y] = 1
	return yh
	
def one_hot_inv(yh):
	"""
	Convert onehot array back to labels
	ex: >>>y_h = np.array([[0,1,0,0],
	    		  [0,0,1,0],
	    		  [0,0,0,1]])
	    >>>y = one_hot_inv(y_h)
	    >>>y
	    np.array([1,2,3])
	"""
	_, inv = np.where(yh == 1)
	return inv

# Step 1. Load Images & Perform Preprocessing
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

X = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
Y = one_hot(train_labels)
X = X/255

Xt = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
Yt = one_hot(test_labels)
Xt = Xt/255

x_size = X.shape[1]


# Define the fully connected network architecture

m = model()
m.add('input', node_number = x_size)
m.add('fully_connected', relu(), 20)
m.add('fully_connected', relu(), 20)
m.add('fully_connected', softmax(), Y.shape[1])
m.initialize_weights(He_initialization)

#Initialize Cost
m.initialize_cost(cost_function = cross_entropy_cost())

#Check if the model performs back propagation correctly
m.gradient_check(X[0], Y[0])

m.Train(X, Y, lr = 0.01, epoch = 50)
m.evaluate(Xt, Yt) 
