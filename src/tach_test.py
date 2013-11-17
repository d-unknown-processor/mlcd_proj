import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random
from mlp import MLP
import sys

root = "../../"
data_path = root + "data/tachfiles/"
sirs_root = data_path
sepshock_root = data_path
sevsep_root = data_path
train_map = root + "tiny_trainset.recs"

def get_data():
  count = 277549643
  mean = 50.1555896038
  stdev = 43.9111167831
  mx = 279.822
  mn = 6.66125e-05

  training_file = ""
  training = []
  dev = []
  f = open(train_map, 'r').readlines()
  for line in f:
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 2:
      continue
    data = None
    cl = None
    stop_point = int(toks[1])
    try:
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".tach", 'r')
        data = f.readlines()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".tach", 'r')
        data = f.readlines()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".tach", 'r')
        data = f.readlines()
        f.close()
        cl = 1
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
    training_x = []
    # We should stop once the disease is acquired, since
    # treatment might come afterwards.
    for l in data[:stop_point]:
      if "Inf" in l or "NaN" in l:
        print "Has Infs or NaNs"
      if len(l) < 1:
        continue
      raw_x = float(l)
      if raw_x == "-Inf":
        xs.append(mn)
      elif raw_x == "Inf":
        xs.append(mx)
      elif raw_x == "NaN":
        xs.append(mx)
      else:
        training_x.append((raw_x - mean)/stdev)
    training.append((training_x, cl))
  f.close()
  return training


def train(mlp, xs, y, window_size, window_step_size):
  i = window_size
  grad = None
  while i < len(xs):
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
  mlp.update_weights(grad)

def train_online(mlp, xs, y, window_size, window_step_size):
  i = window_size
  while i < len(xs):
    x = xs[i - window_size: i]
    X = np.array(x)
    output = mlp.forwards(X)
    T = np.array(y)
    mlp.backwards_online(T, X)
    i += window_step_size


def loss(mlp, data_set, window_size, window_step_size):
  total_loss = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs):
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      print output
      if y == 1:
        total_loss -= np.log(output)
      if y == 0:
        total_loss -= np.log(1-output)
      i += window_step_size
  return total_loss
 
def error(mlp, data_set, window_size, window_step_size):
  class1_error = 0
  class0_error = 0
  count0 = 0
  count1 = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs):
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      print "t=" + str(i) + " p(1|xs) = " + str(output) + " correct label = " + str(y)
      if y == 1:
        class1_error += math.fabs(output[0] - y)
        count1 += 1
      if y == 0:
        class0_error += math.fabs(output[0] - y)
        count0 += 1
      i += window_step_size
  print "class0 error " + str(class0_error/float(count0))
  print "class1 error " + str(class1_error/float(count1))

def main():
  training = get_data()
  # RM this:
  new_training = []
  c1 = 0
  c0 = 0
  for xs, y in training:
    if y == 1 and c1 < 2:
      new_training.append((xs, y))
      c1 +=1
    if y == 0 and c0 < 2:
      new_training.append((xs, y))
      c0 +=1
  training = new_training
  #
  print "len training = " + str(len(training))
  count0 = 0
  count1 = 0
  random.shuffle(training)
  window_size = 7000
  n_input = window_size
  n_hidden = 100
  n_output = 1
  num_hidden_layers = 1
  eta = .0001
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output, eta)
  n_epochs = 10

  l = loss(mlp, training, window_size, window_size/2)
  print "initial loss: " + str(l)
  for j in range(0, n_epochs):
    print "epoch " + str(j)
    random.shuffle(training)
    c = 0
    for xs, y in training:
      if c == 5:
        break
      print "training data " + str(c)
      c += 1
      train(mlp, xs, y, window_size, window_size/2)
      l = loss(mlp, training, window_size, window_size/2)
      print "loss: " + str(l)

  print "Getting training accuracy..."
  error(mlp, training, window_size, window_size/2)

if __name__ == "__main__":
  main()

