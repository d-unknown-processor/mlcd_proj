import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random

class Layer:
  def __init__(self, n_input, n_output, id):
    self.n_input = n_input
    self.n_output = n_output
    self.W = self.init_weights()
    self.X = None # Input
    self.id = id 

  def init_weights(self):
    weights = []
    for i in range(self.n_input + 1):
      weights.append(np.random.uniform(-1, 1, size=(self.n_output)))
    return np.array(weights)

  # Here g is a logistic function
  def g(self, x):
   return 1.0/(1.0 + math.exp(-x))
   
  def get_output(self, X):
    X = np.append(np.array(X, copy=True), 1)
    self.X = X
    activation = np.vectorize(self.g)
    A = X.dot(self.W)
    self.output = activation(A)
    return self.output

  def update_weights(self, eta, d_o):
    W = []
    for i in range(0, len(self.W)):
      w = []
      for j in range(0, len(self.W[i])):
        w.append(self.W[i][j] + eta * d_o[j] * self.X[i])
      W.append(w)
    self.W = np.array(W)
    print str(self.id) + " W: " + str(self.W[0][0])

  def get_error(self, T):
    # Compute error and derivative d_o at output
    d_o = T - self.output
    for i in range(0, len(d_o)):
      d_o[i] *= self.output[i] * (1 - self.output[i])
    return d_o
  
  # V is the weight matrix for the layer which succeeds this one
  def compute_derivative(self, d_o, V):
    # Compute error and derivative d_h at hidden
    d_h = np.array(self.output, copy=True)
    for i in range(0, len(d_h)):
      v = 0
      for j in range(0, len(d_o)):
        v += V[i][j] * d_o[j]
      d_h[i] *= (1 - self.output[i]) * v
    return d_h
   

class MLP:
  def __init__(self, n_input, num_hidden_layers, n_hidden, n_output, eta=1):
    self.n_input = n_input
    self.n_num_hidden_layers = num_hidden_layers
    self.n_output = n_output
    self.eta = eta
    
    self.layers = []
    self.layers.append(Layer(n_input, n_hidden, 0))
    for i in range(1, num_hidden_layers):
      self.layers.append(Layer(n_hidden, n_hidden, i))
    self.output_layer = Layer(n_hidden, n_output, num_hidden_layers)

  def forwards(self, X):
    #print "X at beginning of forwards = " + str(X)
    # Propogate input through each layer
    for layer in self.layers:
      X = layer.get_output(X)
      #print "X in forwards loop = " + str(X)
    O = self.output_layer.get_output(X)
    #print "O = " + str(O)

  def backwards(self, T, X):
    d_o = self.output_layer.get_error(T)
    self.output_layer.update_weights(self.eta, d_o)
    prev_layer = self.output_layer
    X = np.append(X, 1)
    for i in range(1, len(self.layers)+1):
      idx = len(self.layers) - i
      d_o = self.layers[idx].compute_derivative(d_o, prev_layer.W)
      prev_layer = self.layers[idx-1]
      self.layers[idx].update_weights(self.eta, d_o)
