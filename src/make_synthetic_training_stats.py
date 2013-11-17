import numpy as np
import sys
import math

root = "../../"
corrupted_root = root + "fake_features/corrupted/"
uncorrupted_root = root + "fake_features/uncorrupted/"
train_map = root + "fake_trainset.recs"

def get_mean():
  count = 0
  sm = 0
  mx = None
  mn = None
  f = open(train_map, 'r').readlines()
  for line in f:
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 2:
      continue
    data = None
    try:
      if toks[1] == "0": 
        f = open(uncorrupted_root + toks[0], 'r')
        data = f.read()
        f.close()
      elif toks[1] == "1":
        f = open(corrupted_root + toks[0], 'r')
        data = f.read()
        f.close()
    except Exception:
      print "missing file"
    for l in data.split('\n'):
      if "Inf" in l or "NaN" in l:
        continue
      if len(l) < 1:
        continue
      feature = float(l)
      count += 1
      sm += feature
      if mx == None:
        mx = feature
      else:
        if mx < feature:
          mx = feature
      if mn == None:
        mn = feature
      else:
        if mn > feature:
          mn = feature
  return sm/float(count), mx, mn, count


def get_stdev(mean, count):
  squared_diff = 0
  f = open(train_map, 'r')
  for line in f:
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 2:
      continue
    data = None
    try:
      if toks[1] == "0": 
        f = open(uncorrupted_root + toks[0], 'r')
        data = f.read()
        f.close()
      elif toks[1] == "1":
        f = open(corrupted_root + toks[0], 'r')
        data = f.read()
        f.close()
    except Exception:
      print "missing file"
    for l in data.split('\n'):
      if len(l) < 1:
        continue
      feature = float(l)
      squared_diff += (feature - mean)**2
  return math.sqrt(squared_diff / float(count))
    
def main():
  mean, mx, mn, count = get_mean()
  stdev = get_stdev(mean, count)
  print "count = " + str(count)
  print "mean = " + str(mean) 
  print "stdev = " + str(stdev)
  print "max = " + str(mx)
  print "min = " + str(mn)


if __name__ == "__main__":
  main()
