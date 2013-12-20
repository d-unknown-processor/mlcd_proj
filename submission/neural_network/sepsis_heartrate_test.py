# Author: David Snyder
#
# Main script for current sepsis experiments with heart rate data.
# Unfortunately, it is impractical to include the gigs of data in
# this submission, therefore this script will not run for you.
# However, I'm happy to demo this in person.
# To simply demo neural network performance, I've included a
# script called density_demo.py, which will run.
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
dev_map = root + "devset.recs"

# Get training or dev data
def get_data(window_size, dev=False):
# These statistics are necessary to 
# normalize the data:
#MEAN
  m_count = 7872693
  m_mean = 51.1187719551
  m_stdev = 42.584992022
  m_max = 215.017428333
  m_min = 6.66125e-05

#STDEV
  s_count = 7872693
  s_mean = 3.81859661117
  s_stdev = 6.61712470069
  s_max = 108.463107754
  s_min = 0.0

  training_file = ""
  training = []
  training_c1 = []
  training_c0 = []

  count_c0 = 0
  count_c1 = 0
  if dev:
    f = open(dev_map, 'r').readlines()
  else:
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
        f = open(sirs_root + toks[0] + ".f3", 'r')
        data = f.readlines()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".f3", 'r')
        data = f.readlines()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".f3", 'r')
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
    for l in data[stop_point - window_size -1 : stop_point]:
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
    if cl == 1:
      training_c1.append((training_x, cl))
    if cl == 0:
      training_c0.append((training_x, cl))
  f.close()
  training_c0_new = []
  training_c1_new = []
  for i in range(len(training_c0)):
    xs, y = training_c0[i]
    if len(xs) > window_size:
      training_c0_new.append((xs,y))
  for i in range(len(training_c1)):
    xs, y = training_c1[i]
    if len(xs) > window_size:
      training_c1_new.append((xs,y))

  random.shuffle(training_c0_new)
  random.shuffle(training_c1_new)
  
  if len(training_c0_new) > len(training_c1_new):
    training_c0_new = training_c0_new[:len(training_c1_new)]
  if len(training_c1_new) > len(training_c0_new):
    training_c1_new = training_c1_new[:len(training_c0_new)]

  training.extend(training_c1_new)
  training.extend(training_c0_new)
  random.shuffle(training)
  
  return training, count_c0, count_c1

def get_time_series():
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
  series_c0 = []
  series_c1 = []
  count_c0 = 0
  count_c1 = 0
  f = open(dev_map, 'r').readlines()
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
        f = open(sirs_root + toks[0] + ".f3", 'r')
        data = f.readlines()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".f3", 'r')
        data = f.readlines()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".f3", 'r')
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
    if cl == 1:
      series_c1.append((training_x, cl))
    if cl == 0:
      series_c0.append((training_x, cl))
  random.shuffle(series_c0)
  random.shuffle(series_c1)
  c0_limit = 10
  if len(series_c0) < 10:
    c0_limit = len(series_c0)
  c1_limit = 10
  if len(series_c1) < 10:
    c1_limit = len(series_c1)

  series_c0 = series_c0[0:c0_limit]
  series_c1 = series_c1[0:c1_limit]
  return series_c0, series_c1

# Train using standard batch gradient descent.
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
  return grad

# Train using a form of stochastic gradient descent. This isn't
# currently used.
def train_online(mlp, xs, y, window_size, window_step_size):
  i = window_size
  while i < len(xs):
    x = xs[i - window_size: i]
    X = np.array(x)
    output = mlp.forwards(X)
    T = np.array(y)
    mlp.backwards_online(T, X)
    i += window_step_size


# Calculate the cross entropy loss on the training data.
def loss(mlp, data_set, window_size, window_step_size):
  total_loss = 0
  count = 0
  for xs, y in data_set:
    i = window_size
    while i < len(xs):
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
 
# Calculate two kinds of error rates: error rate on raw probability
# and error rate after applying the signum function. The latter 
# should be viewed as the more important number.
def error(mlp, data_set, window_size, window_step_size, dev=False):
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
  if not dev:
    print "train class0 error " + str(class0_true_error/float(count0)) + ", " + str(class0_error/float(count0))
    print "train class1 error " + str(class1_true_error/float(count1)) + ", " + str(class1_error/float(count1))
    print "train total error " + str((class1_true_error + class0_true_error)/float(count1 + count0)) + ", " + str((class1_error + class0_error)/float(count1 + count0))
  if dev:
    print "dev class0 error " + str(class0_true_error/float(count0)) + ", " + str(class0_error/float(count0))
    print "dev class1 error " + str(class1_true_error/float(count1)) + ", " + str(class1_error/float(count1))
    print "dev total error " + str((class1_true_error + class0_true_error)/float(count1 + count0)) + ", " + str((class1_error + class0_error)/float(count1 + count0))


# Calculate probability of developing sepsis with a sliding window
# over class 0 (SIRS) and class 1 (Severe Sepsis and Septic Shock).
def probs(mlp, series_c0, series_c1, window_size, window_step_size):
  print "Get probabilities..." 
  class1_true_error = 0
  class0_true_error = 0
  class1_error = 0
  class0_error = 0
  count0 = 0
  count1 = 0
  print "c0 probs"
  for xs, y in series_c0:
    i = window_size
    while i < len(xs):
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      print output[0], math.fabs(output[0] - y)
      i += window_step_size
    print
  print "c1 probs"
  for xs, y in series_c1:
    i = window_size
    while i < len(xs):
      x = xs[i - window_size:i]
      X = np.array(x)
      output = mlp.forwards(X)
      print output[0], math.fabs(output[0] - y)
      i += window_step_size
    print

# Force a decision from the network. If the probability of sepsis is greater
# than or equal to 0.5, output 1, otherwise output 0.
def sgn(x):
  if x > .5:
    return 1.0
  else:
    return 0.0

# A helper function for summing partial gradients together.
def sum_grad(grad, part_grad):
  if part_grad != None:
    if grad == None:
      grad = part_grad
    else:
      for l in range(0, len(part_grad)):
        for k in range(0, len(part_grad[l])):
          grad[l][k] += part_grad[l][k]

  return grad

# Main loop for training
def main():
  training, count_c0, count_c1 = get_data(122)
  dev, _, _ = get_data(122,True)
  series_c0, series_c1 = get_time_series()
  # A window of 240 corresponds to two hours of heart rate
  window_size = 240
  print "#class0: ", count_c0/float(window_size), "#class1:", count_c1/float(window_size)
  c1 = 0
  c0 = 0
  print "len training = " + str(len(training))
  count0 = 0
  count1 = 0
  random.shuffle(training)
  n_input = window_size
  n_hidden = 100
  n_output = 1
  num_hidden_layers = 1
  eta_c = .005
  A = .0008
  eta = eta_c
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output, eta)
  n_epochs = 400
  step = True

  l = loss(mlp, training, window_size, window_size/2)
  print "initial loss: " + str(l)
  for j in range(0, n_epochs):
    print "epoch " + str(j)
    random.shuffle(training)
    c = 0
    grad = None
    for xs, y in training:
      if c == 10:
        break
      c += 1
      part_grad = None
      if step:
        part_grad = train(mlp, xs, y, window_size, window_size/2)
      else:
        part_grad = train(mlp, xs, y, window_size, 1)

      grad = sum_grad(grad, part_grad)
    mlp.update_weights(grad)
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
    if n_epochs % 10 == 0:
      if step:
        error(mlp, dev, window_size, window_size/2, True)
      else:
        error(mlp, dev, window_size, 1, True) 

  print "Getting training accuracy..." 
  if step:
    error(mlp, training, window_size, window_size/2)
  else:
    error(mlp, training, window_size, 1)
  print "Getting dev accuracy..." 
  if step:
    error(mlp, dev, window_size, window_size/2, True)
    probs(mlp, series_c0, series_c1, window_size, window_size/2)
  else:
    error(mlp, dev, window_size, 1, True)
    probs(mlp, series_c0, series_c1, window_size, 1)


if __name__ == "__main__":
  main()

