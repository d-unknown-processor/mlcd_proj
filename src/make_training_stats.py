import numpy as np

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
train_map = root + "trainset.recs"

def get_mean():
  count = 0
  sums = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  f = open(train_map, 'r')
  for line in f:
    toks = line.split()
    if len(toks) < 3:
      continue
    data = None
    if toks[2] == "4": # sirs
      f = open(sirs_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
      cl = 0
    elif toks[2] == "3": # septic shock
      f = open(sepshock_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
      cl = 1
    elif toks[2] == "1": # severe sepsis
      f = open(sevsep_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
    else:
      print "Unidentified class"
    for l in data.split('\n'):
      features = [float(x) for x in l.split()]
      if len(features) < 1:
        continue
      count += 1
      for j in range(0, len(features)):
        sums["f" + str(j)] += features[j]
 
  means = []
  for sm in sums.values():
    means.append(sm / float(count))
  return means, count


def get_variance(means, count):
  squared_diffs = {'f0' : 0, 'f1': 0, 'f2' : 0, 'f3' : 0, 'f4' : 0, 'f5' : 0, 'f6' : 0, 'f7':0, 'f8' : 0}
  f = open(train_map, 'r')
  for line in f:
    toks = line.split()
    if len(toks) < 3:
      continue
    data = None
    if toks[2] == "4": # sirs
      f = open(sirs_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
      cl = 0
    elif toks[2] == "3": # septic shock
      f = open(sepshock_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
      cl = 1
    elif toks[2] == "1": # severe sepsis
      f = open(sevsep_root + toks[0] + ".feats", 'r')
      data = f.read()
      f.close()
    else:
      print "Unidentified class"
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
