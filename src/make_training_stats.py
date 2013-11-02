import numpy as np
import sys

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
train_map = root + "trainset.recs"
#{'f0': 174650769.85940018, 'f1': 18772716.698753968, 'f2': 5814838386936.324, 'f3': 6666563201149.626, 'f4': 87834758.10096925, 'f5': 39286188.27412433, 'f6': 31308445.88732576, 'f7': 27458640.814931396, 'f8': 27496105.34393987} 61453018

def get_mean():
  count = 0
  sums = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  sums = {'f0': 174650769.85940018, 'f1': 18772716.698753968, 'f2': 5814838386936.324, 'f3': 6666563201149.626, 'f4': 87834758.10096925, 'f5': 39286188.27412433, 'f6': 31308445.88732576, 'f7': 27458640.814931396, 'f8': 27496105.34393987} 
  count = 61453018
  f = open(train_map, 'r').readlines()
  for line in f[138:]:
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
        sums["f" + str(j)] += features[j]
    print sums, count
 
  means = []
  for sm in sums.values():
    means.append(sm / float(count))
  return means, count


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
        squared_diffs["f" + str(j)] = (features[j] - means[j])**2
  variances = []
  for d in squared_diffs.values():
    variances.append(d / float(count))
  return variances
    
def main():
  means, count = get_mean()
  variances = get_variance(means, count)
  print "count = " + str(count)
  print "means = " + str(means) 
  print "var = " + str(variances)


if __name__ == "__main__":
  main()
