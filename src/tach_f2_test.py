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
train_map = root + "balanced_trainset.recs"
#train_map = root + "tiny_trainset.recs"

def get_data():
#MEAN
  m_count = 182781
  m_mean = 82.6853176832
  m_stdev = 21.8662402705
  m_max = 189.299067222
  m_min = 20.0003154611

#STDEV
  s_count = 182781
  s_mean = 7.98194511242
  s_stdev = 9.12350064717
  s_max = 85.3576842129
  s_min = 0.0873494789783

  training_file = ""
  training = []
  dev = []
  count_c0 = 0
  count_c1 = 0
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
        f = open(sirs_root + toks[0] + ".f2", 'r')
        data = f.readlines()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".f2", 'r')
        data = f.readlines()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".f2", 'r')
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
    if data == None:
      continue
    if stop_point > len(data):
      stop_point = len(data) - 1
    for l in data[:stop_point]:
      if "Inf" in l or "NaN" in l:
        print "Has Infs or NaNs"
      if len(l) < 1:
        continue
      if cl == 1:
        count_c1 += 1
      elif cl == 0:
        count_c0 += 1
    
      raw_xs = l.split(" ")
      raw_m = float(raw_xs[0])
      raw_s = float(raw_xs[1])
      if raw_m == "-Inf":
        xs.append(m_mn)
      elif raw_m == "Inf":
        xs.append(m_mx)
      elif raw_m == "NaN":
        xs.append(m_mx)
      else:
        training_x.append((raw_m - m_mean)/m_stdev)
      if raw_s == "-Inf":
        xs.append(s_mn)
      elif raw_s == "Inf":
        xs.append(s_mx)
      elif raw_s == "NaN":
        xs.append(s_mx)
      else:
        training_x.append((raw_s - s_mean)/s_stdev)
    training.append((training_x, cl))
  f.close()
  return training, count_c0, count_c1


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
  # If grad is None, then we haven't gotten into the while loop,
  # and the length of the input is shorter than the width of the
  # window.
  if grad == None:
    return  
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
  count = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs):
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      #print output
      if y == 1:
        total_loss -= np.log(output)
      if y == 0:
        total_loss -= np.log(1-output)
      i += window_step_size
      count += 1
  # Per frame loss
  return total_loss/float(count)
 
def error(mlp, data_set, window_size, window_step_size):
  class1_true_error = 0
  class0_true_error = 0
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
      #print "t=" + str(i) + " p(1|xs) = " + str(output) + " correct label = " + str(y)
      if y == 1:
        class1_true_error += math.fabs(output[0] - y)
        class1_error += math.fabs(sgn(output[0]) - y)
        count1 += 1
      if y == 0:
        class0_true_error += math.fabs(output[0] - y)
        class0_error += math.fabs(sgn(output[0]) - y)
        count0 += 1
      i += window_step_size
  print "class0 error " + str(class0_true_error/float(count0)) + ", " + str(class0_error/float(count0))
  print "class1 error " + str(class1_true_error/float(count1)) + ", " + str(class1_error/float(count1))
  print "total error " + str((class1_true_error + class0_true_error)/float(count1 + count0)) + ", " + str((class1_error + class0_error)/float(count1 + count0))

def sgn(x):
  if x > .5:
    return 1.0
  else:
    return 0.0

def main():
  training, count_c0, count_c1 = get_data()
  window_size = 20
  print "#class0: ", count_c0/float(window_size), "#class1:", count_c1/float(window_size)
  # RM this:
  new_training = []
  c1 = 0
  c0 = 0
  #training = new_training
  #
  print "len training = " + str(len(training))
  count0 = 0
  count1 = 0
  random.shuffle(training)
  n_input = window_size
  n_hidden = 20
  n_output = 1
  num_hidden_layers = 2
  eta_c = .0005
  A = .0008
  eta = eta_c
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output, eta)
  n_epochs = 1000
  step = True
  window_step = window_size

  l = loss(mlp, training, window_size, window_step)
  print "initial loss: " + str(l)
  for j in range(0, n_epochs):
    print "epoch " + str(j)
    random.shuffle(training)
    c = 0
    for xs, y in training:
      if c == 10:
        break
    #  print "training data " + str(c)
      c += 1
      if step:
        train(mlp, xs, y, window_size, window_step)
      else:
        train(mlp, xs, y, window_size, 1)
    #if j % 5 == 0:
    if step:
      error(mlp, training, window_size, window_step)
    else:
      error(mlp, training, window_size, 1) 
    if step:
      l = loss(mlp, training, window_size, window_step)
    else:
      l = loss(mlp, training, window_size, 1)
    print "loss: " + str(l)
    # eta = A / float(j + 1)
    eta = A / float(j/float(n_epochs) + 1)
    mlp.eta = eta
    print "lr:", mlp.eta

  print "Getting training accuracy..." 
  if step:
    error(mlp, training, window_size, window_step)
  else:
    error(mlp, training, window_size, 1)

if __name__ == "__main__":
  main()

