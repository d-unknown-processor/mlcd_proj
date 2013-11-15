import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random
from mlp import MLP
import sys

root = "../../"
corrupted_root = root + "fake_features/corrupted/"
uncorrupted_root = root + "fake_features/uncorrupted/"
train_map = root + "fake_trainset.recs"

def get_data():
  mean = 8.45626800736
  stdev = 6.50153035644
  mx = 30.010317495
  mn = -0.0314087493786

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
    try:
      if toks[1] == "0": 
        f = open(uncorrupted_root + toks[0], 'r')
        data = f.read()
        cl = 0
        f.close()
      elif toks[1] == "1":
        f = open(corrupted_root + toks[0], 'r')
        data = f.read()
        cl = 1
        f.close()
    except Exception:
      print "missing file"

    training_x = []
    for l in data.split('\n'):
      if "Inf" in l or "NaN" in l:
        print "Has Infs or NaNs"
      if len(l) < 1:
        continue
      raw_x = float(l)
      if raw_x == "-Inf":
        xs.append(-1.1)
      elif raw_x == "Inf":
        xs.append(1.1)
      elif raw_x == "NaN":
        xs.append(0.0)
      else:
        training_x.append((raw_x - mean)/stdev)
    training.append((training_x, cl))
  f.close()
  return training

def main():
  training = get_data()
  mn = None
  for xs, y in training:
    if mn == None:
      mn = len(xs)
    elif len(xs) < mn:
      mn = len(xs)
  dev = []
  count0 = 0
  count1 = 0
  mx = 3
  idxs_to_delete = []
  random.shuffle(training)
  for i in range(0, len(training)):
    x, y = training[i]
    if count0 >= mx and count1 >= mx:
      break
    if y == 0 and count0 <= mx:
      dev.append((x,y))
      count0 += 1
      idxs_to_delete.append(i)
    if y == 1 and count1 <= mx:
      dev.append((x,y))
      count1 += 1
      idxs_to_delete.append(i)
  for i in range(0, len(idxs_to_delete)):
    del(training[i])
  
  window_size = 100
  n_input = window_size
  n_hidden = 20
  n_output = 1
  num_hidden_layers = 1
  eta = 0.2
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output, eta)
  n_epochs = 100
  for j in range(0, n_epochs):
    class1_error_total = 0
    class0_error_total = 0
    class0_count = 0
    class1_count = 0
    random.shuffle(training)
    for xs, y in training:
      class0_error = 0
      class1_error = 0
      count = 0
      xs = xs[0:mn]
      i = window_size
      while i < len(xs):
        x = xs[i - window_size: i]
        X = np.array(x)
        output = mlp.forwards(X)
        T = np.array(y)
        mlp.backwards(T, X)
        if y == 1:
          class1_error += math.fabs(output[0] - y)
        if y == 0:
          class0_error += math.fabs(output[0] - y)
        count += 1
        i += window_size/2
      if y == 0:
        class0_count += 1
        class0_error_total += (class0_error/float(count))
#        print "class 0 error: " + str(class0_error/float(count))
      if y == 1:
        class1_count += 1
        class1_error_total += (class1_error/float(count))
#        print "class 1 error: " + str(class1_error/float(count))
    print "epoch: " + str(j) + " class0_err: " + str(class0_error_total/float(class0_count)) + " class1_err: " + str(class1_error_total/float(class1_count))
  class1_error = 0
  class0_error = 0
  count0 = 0
  count1 = 0
  for xs, y in dev:
   x = xs[i:i+window_size]
   X = np.array(x)
   for i in range(0,len(xs)-window_size):
      x = xs[i:i+window_size]
      X = np.array(x)
      output = mlp.forwards(X)
      print "t=" + str(i) + " p(1|xs) = " + str(output) + " correct label = " + str(y)
      if y == 1:
        class1_error += math.fabs(output[0] - y)
        count1 += 1
      if y == 0:
        class0_error += math.fabs(output[0] - y)
        count0 += 1
  print "class0 error " + str(class0_error/float(count0))
  print "class1 error " + str(class1_error/float(count1))
      

if __name__ == "__main__":
  main()

