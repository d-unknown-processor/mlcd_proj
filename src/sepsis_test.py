import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math
import random
from mlp import MLP
import sys

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
train_map = root + "tiny_trainset.recs"

def get_data():
  means = [2.8721788087662903, 0.2806617953787631, 81852.17209332075, 115494.86388216066, 1.4323046825936003, 0.6144981211588969, 0.49158286796476647, 0.4388657908165723, 0.4406640872660855]
  variances = [6.311123507845681e-10, 8.245893870741903e-09, 42.44463007381666, 86.74717813612696, 2.813928413273801e-09, 1.8732585384841253e-09, 9.083890232598939e-11, 1.218964651908642e-11, 4.289569892421104e-11]
  training_file = ""
  training = []
  dev = []
  fi = open(train_map, 'r').readlines()
  for l in fi:
    print "l in fi = " + str(l)
    sys.stderr.write(".")
    toks = l.split()
    if len(toks) < 3:
      continue
    data = None
    cl = None
    try:
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 1
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 4
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 3
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    training_x = []
    for l in data.split('\n'):
      if "Inf" in l or "NaN" in l:
        print "Has Infs or NaNs"
      xs = []
      raw_xs = l.split(' ')
      if len(raw_xs) < 9:
        continue
      for i in range(0, len(raw_xs)):
        if raw_xs[i] == "-Inf":
          xs.append(-1.1)
        elif raw_xs[i] == "Inf":
          xs.append(1.1)
        elif raw_xs[i] == "NaN":
          xs.append(0.0)
        else:
          x = float(raw_xs[i])
          xs.append((x - means[i])/math.sqrt(variances[i]))
      print xs
      training_x.append((xs, cl))
    training.append(training_x)
  f.close()
  print training
  return training

def main():
  training = get_data()
  n_input = 9
  n_hidden = 6
  n_output = 1
  num_hidden_layers = 3
  mlp = MLP(n_input, num_hidden_layers, n_hidden, n_output)
  n_epochs = 100
  for i in range(0, n_epochs):
    random.shuffle(training)
    for xs, y in training: 
      X = np.array(xs)
      mlp.forwards(X)
      T = np.array(y)
      mlp.backwards(T, X)

if __name__ == "__main__":
  main()

