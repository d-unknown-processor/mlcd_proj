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

  # Initialize weights to small values.
  # I'm samplying from uniform(0, 1).
  # There is probably a smarter way to do this.
  def init_weights(self):
    weights = []
    for i in range(self.n_input + 1):
      weights.append(np.random.normal(0, 1.0/math.sqrt(self.n_input + 1), size=(self.n_output)))
    return np.array(weights)

  # Here g is a logistic function
  def g(self, x):
   try:
     return 1.0/(1.0 + math.exp(-x))
   except Exception:
     print "In overflow x = " + str(x)
     return 1.0
   
  # Get output from any layer
  def get_output(self, X):
    X = np.append(np.array(X, copy=True), 1)
    self.X = X
    activation = np.vectorize(self.g)
    A = X.dot(self.W)
    self.output = activation(A)
    return self.output

  # For updating weights after each training example.
  def update_weights(self, gradient):
    W = []
    for i in range(0, len(self.W)):
      w = []
      for j in range(0, len(self.W[i])):
        w.append(self.W[i][j] + gradient[i][j])
      W.append(w)
    self.W = np.array(W)

  # For updating weights after each window in the training example.
  def update_weights_online(self, gradient):
    W = []
    for i in range(0, len(self.W)):
      w = []
      for j in range(0, len(self.W[i])):
        w.append(self.W[i][j] + gradient[j] * self.X[i])
      W.append(w)
    self.W = np.array(W)

  # Get output layer gradient using 
  def get_output_layer_gradient_cross_ent(self, T, eta):
    d_o = T - self.output # Error
    for i in range(0, len(d_o)):
      d_o[i] *= eta
    return d_o

  def get_output_layer_gradient_squared_err(self, T, eta):
    d_o = T - self.output # Error
    for i in range(0, len(d_o)):
      d_o[i] *= self.output[i] * (1 - self.output[i]) * eta
    return d_o
  
  # V is the weight matrix for the layer which succeeds this one
  def get_gradient(self, d_o, V, eta):
    # Compute error and derivative d_h at hidden
    d_h = np.array(self.output, copy=True)
    for i in range(0, len(d_h)):
      v = 0
      for j in range(0, len(d_o)):
        v += V[i][j] * d_o[j]
      d_h[i] *= eta * (1 - self.output[i]) * v
    return d_h
   
  # To be added to the weight
  def get_weight_update(self, gradient):
    W = []
    for i in range(0, len(self.W)):
      w = []
      for j in range(0, len(self.W[i])):
        w.append(gradient[j] * self.X[i])
      W.append(w)
    return np.array(W)

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

  # Propogate input through each layer
  def forwards(self, X):
    for layer in self.layers:
      X = layer.get_output(X)
    return self.output_layer.get_output(X)

  # Update the weights given a gradient
  def update_weights(self, grad):
    self.output_layer.update_weights(grad[0])
    for i in range(1, len(self.layers)+1):
      idx = len(self.layers) - i
      self.layers[idx].update_weights(grad[i])

  # Get the gradient for each layer
  def backwards(self, T, X):
    d_o = self.output_layer.get_output_layer_gradient_cross_ent(T, self.eta)
    grad = [self.output_layer.get_weight_update(d_o)]
    prev_layer = self.output_layer
    X = np.append(X, 1)
    for i in range(1, len(self.layers)+1):
      idx = len(self.layers) - i
      d_o = self.layers[idx].get_gradient(d_o, prev_layer.W, self.eta)
      prev_layer = self.layers[idx-1]
      grad.append(self.layers[idx].get_weight_update(d_o))
    return grad

  def backwards_online(self, T, X):
    d_o = self.output_layer.get_output_layer_gradient_cross_ent(T, self.eta)
    self.output_layer.update_weights_online(d_o)
    prev_layer = self.output_layer
    X = np.append(X, 1)
    for i in range(1, len(self.layers)+1):
      idx = len(self.layers) - i
      d_o = self.layers[idx].get_gradient(d_o, prev_layer.W, self.eta)
      prev_layer = self.layers[idx-1]
      self.layers[idx].update_weights_online(d_o)
