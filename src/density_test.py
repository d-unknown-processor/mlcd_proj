import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random
from mlp import MLP

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

def main():
  training, dev = get_data()
  n_input = 5
  n_hidden = 6
  n_output = 1
  num_hidden_layers = 3
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output)
  n_epochs = 1000
  # For testing that learning happens...
  # with a toy function eg, f(1,0) = 0.
  for i in range(0, n_epochs):
    random.shuffle(training)
    for xs, y in training: 
      X = np.array(xs)
      mlp.forwards(X)
      T = np.array(y)
      mlp.backwards(T, X)

if __name__ == "__main__":
  main()

