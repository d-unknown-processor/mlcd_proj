import numpy as np
import sys
import math

root = "../../"
data_path = root + "data/tachfiles/"
sirs_root = data_path
sepshock_root = data_path
sevsep_root = data_path
train_map = root + "trainset.recs"

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
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    if len(data.split('\n')) < 10:
      continue
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
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".tach", 'r')
        data = f.read()
        f.close()
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    if len(data.split('\n')) < 10:
      continue
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
