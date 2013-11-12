import numpy as np
import sys

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
#train_map = root + "trainset.recs"
train_map = root + "trainset.recs"

def get_mean():
  count = 0
  sums = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  maxs = {'f0' : None, 'f1': None, 'f2' : None, 'f3' : None, 'f4' : None, 'f5' : None, 'f6' : None, 'f7':None, 'f8' : None}
  mins = {'f0' : None, 'f1': None, 'f2' : None, 'f3' : None, 'f4' : None, 'f5' : None, 'f6' : None, 'f7':None, 'f8' : None}
  f = open(train_map, 'r').readlines()
  for line in f:
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 3:
      continue
    data = None
    try:
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    for l in data.split('\n'):
      if "Inf" in l or "NaN" in l:
        continue
      features = [float(x) for x in l.split()]
      if len(features) < 1:
        continue
      count += 1
      for j in range(0, len(features)):
        k = "f" + str(j)
        sums[k] += features[j]
        if maxs[k] == None:
          maxs[k] = features[j]
        else:
          if maxs[k] < features[j]:
            maxs[k] = features[j]
        if mins[k] == None:
          mins[k] = features[j]
        else:
          if mins[k] > features[j]:
            mins[k] = features[j]
  means = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  for k in sums:
    means[k] = (sums[k] / float(count))
  return means, maxs, mins, count


def get_variance(means, count):
  squared_diffs = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  f = open(train_map, 'r')
  for line in f:
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 3:
      continue
    data = None
    try:
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 0
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
        cl = 1
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".feats", 'r')
        data = f.read()
        f.close()
      else:
        sys.stderr.write("Unidentified class")
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    for l in data.split('\n'):
      features = [float(x) for x in l.split()]
      if len(features) < 1:
        continue
      for j in range(0, len(features)):
        squared_diffs["f" + str(j)] = (features[j] - means["f" + str(j)])**2
  variances = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  for k in squared_diffs:
    variances[k] = (squared_diffs[k] / float(count))
  return variances
    
def main():
  means, maxs, mins, count = get_mean()
  variances = get_variance(means, count)
  print "count = " + str(count)
  print "means = " + str(means) 
  print "var = " + str(variances)
  print "maxs = " + str(maxs)
  print "mins = " + str(mins)


if __name__ == "__main__":
  main()
