# Author: David Snyder
#
# A script for calculating mean and standard deviation over one minute intervals.
# These statistics serve as features for the neural network. 
# Unfortunately you won't be able to run
# this, since it's impractical to include the training data. I'm happy
# to demo this in person if necessary.
import numpy as np
import sys
import math

root = "../../"
data_path = root + "data/tachfiles/"
sirs_root = data_path
sepshock_root = data_path
sevsep_root = data_path
train_map = root + "devset.recs"

# Calculate 1 minute means over all heart rate data. 
def make_averaged_data():
  # A one minute window
  window = 120
  f = open(train_map, 'r').readlines()
  for line in f:
    toks = line.split()
    if len(toks) < 2:
      continue
    data = None
    try:
      f2 = open(data_path + toks[0] + ".tach", 'r')
      data = f2.readlines()
      f2.close()
    except Exception:
      continue
    i = 0
    new_data = []
    step_size = window/2
    while i < len(data) - step_size:
      s = [float(x) for x in data[i:i+window]]
      mean = np.mean(s)
      stdev = np.std(s)
      new_data.append((mean, stdev))
      i += step_size
    print new_data
    s = ""
    for x in new_data:
      s += str(x[0]) + " " + str(x[1]) + "\n"
    print s
    f3 = open(data_path + toks[0] + ".f3", 'w')
    f3.write(s)
  
def main():
  make_averaged_data()
if __name__ == "__main__":
  main()
