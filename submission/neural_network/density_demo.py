# Author: David Snyder
#
# All params are modified in the main. There are no command line arguments.
# Usually this set up yield a validation set error rate of about 15%.
# Testing on density classification task as a demo.
#
# The density classification task can be viewed as a majority vote. If
# the input has more 1s than 0s, the correct label is 1. 
# If the input has more 0s than 1s, the label is 0.
# For instance 00011 -> 0 and 10101 -> 1.
import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random
from mlp import MLP

# Load the training and dev data
def get_data():
  training_file = "density_train.txt"
  dev_file = "density_dev.txt"
  training = []
  dev = []
  f = open(training_file, 'r')
  for l in f:
    xs, y = l.split(' ')
    xs = [float(x) for x in xs]
    y = float(y)
    training.append((xs, y))
  f.close()
  f = open(dev_file, 'r')
  for l in f:
    xs, y = l.split(' ')
    xs = [float(x) for x in xs]
    y = float(y)
    dev.append((xs, y))
  f.close()
  return training, dev

# Batch training (using batch gradient descent)
def train(mlp, xs, y, window_size, window_step_size):
  i = window_size
  grad = None
  while i < len(xs) + 1:
    x = xs[i - window_size: i]
    X = np.array(x)
    output = mlp.forwards(X)
    T = np.array(y)
    if grad == None:
      grad = mlp.backwards(T, X)
    else:
      partial_grad = mlp.backwards(T, X)
      for l in range(0, len(partial_grad)):
        for k in range(0, len(partial_grad[l])):
          grad[l][k] += partial_grad[l][k]
    i += window_step_size
  # If grad is None, then we haven't gotten into the while loop,
  # and the length of the input is shorter than the width of the
  # window.
  if grad == None:
    return  
  mlp.update_weights(grad)

# Training method for stochastic gradient descent
def train_online(mlp, xs, y, window_size, window_step_size):
  i = window_size
  while i < len(xs):
    x = xs[i - window_size: i]
    X = np.array(x)
    output = mlp.forwards(X)
    T = np.array(y)
    mlp.backwards_online(T, X)
    i += window_step_size


# calculate the perframe loss. Here the loss function
# is cross entropy.
def loss(mlp, data_set, window_size, window_step_size):
  total_loss = 0
  count = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs) + 1:
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      if y == 1:
        total_loss -= np.log(output)
      if y == 0:
        total_loss -= np.log(1-output)
      i += window_step_size
      count += 1
  # Per frame loss
  return total_loss/float(count)
 
# Calculate the error. This function prints two kinds of error:
# a) error rate on raw probabilities before applying the signum function
# and b) error after applying signum.
def error(mlp, data_set, window_size, window_step_size):
  class1_true_error = 0
  class0_true_error = 0
  class1_error = 0
  class0_error = 0
  count0 = 0
  count1 = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs) + 1:
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      if y > .9:
        class1_true_error += math.fabs(output[0] - y)
        class1_error += math.fabs(sgn(output[0]) - y)
        count1 += 1
      if y < 0.1:
        class0_true_error += math.fabs(output[0] - y)
        class0_error += math.fabs(sgn(output[0]) - y)
        count0 += 1
      i += window_step_size
  print "class0 error " + str(class0_true_error/float(count0)) + ", " + str(class0_error/float(count0))
  print "class1 error " + str(class1_true_error/float(count1)) + ", " + str(class1_error/float(count1))
  print "total error " + str((class1_true_error + class0_true_error)/float(count1 + count0)) + ", " + str((class1_error + class0_error)/float(count1 + count0))

# If the output probability is 
def sgn(x):
  if x > .5:
    return 1.0
  else:
    return 0.0

def main():
  training, dev = get_data()
  window_size = 5
  n_input = window_size
  n_hidden = 100
  n_output = 1
  A = 1
  num_hidden_layers = 1
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output)
  n_epochs = 50
  step = False
  l = loss(mlp, training, window_size, window_size/2)
  print "initial loss: " + str(l)
  for j in range(0, n_epochs):
    print "epoch " + str(j)
    random.shuffle(training)
    c = 0
    for xs, y in training:
      if c == 10:
        break
      c += 1
      if step:
        train(mlp, xs, y, window_size, window_size/2)
      else:
        train(mlp, xs, y, window_size, 1)
    if step:
      error(mlp, training, window_size, window_size/2)
    else:
      error(mlp, training, window_size, 1) 
    if step:
      l = loss(mlp, training, window_size, window_size/2)
    else:
      l = loss(mlp, training, window_size, 1)
    print "loss: " + str(l)
    eta = A / float(j/float(n_epochs) + 1)
    mlp.eta = eta
    print "lr:", mlp.eta

  print "Getting Dev Accuracy..." 
  if step:
    error(mlp, dev, window_size, window_size/2)
  else:
    error(mlp, dev, window_size, 1)

if __name__ == "__main__":
  main()

