import numpy as np
import numpy.ma as ma
import sys

root = "../../"
sepshock_root = root + "features/sepshock/"
sevsep_root = root + "features/sevsep/"
sirs_root = root + "features/sirs/"
train_map = root + "trainset.recs"


def get_mean():
    print 'accumilating means...'
    count = 0
    cumilative_sums = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sums = {'f0': 0, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0}
    f = open(train_map, 'r')

    for line in f:
        sys.stderr.write('.')
        toks = line.split()
        if len(toks) < 3:
            filepath = ''
            continue
            #data = None
        if toks[2] == "1": # sirs
            #f = open(sirs_root + toks[0] + ".feats", 'r')
            #data = f.read()
            #f.close()
            filepath = sirs_root + toks[0] + ".feats"
            #cl = 0
        elif toks[2] == "4": # septic shock
            #f = open(sepshock_root + toks[0] + ".feats", 'r')
            #data = f.read()
            #f.close()
            filepath = sepshock_root + toks[0] + ".feats"
            #cl = 1
        elif toks[2] == "3": # severe sepsis
            #f = open(sevsep_root + toks[0] + ".feats", 'r')
            #data = f.read()
            filepath = sevsep_root + toks[0] + ".feats"
            #f.close()
        else:
            print "Unidentified class"
        if filepath != '':
            dt = np.loadtxt(filepath)
            mdt = ma.masked_invalid(dt)
            sum = np.sum(mdt, axis=0)
            cumilative_sums = cumilative_sums + sum
        '''
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
        '''
        means = cumilative_sums / float(count)
    return list(means), count


def get_variance(means, count):
    print 'accumilating squared diffs'
    squared_diffs = {'f0': 0, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0, 'f7': 0, 'f8': 0}
    cumilative_squared_diff = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    f = open(train_map, 'r')
    for line in f:
        sys.stderr.write('.')
        toks = line.split()
        if len(toks) < 3:
            filepath = ''
            continue
            #data = None
        if toks[2] == "1": # sirs
            #f = open(sirs_root + toks[0] + ".feats", 'r')
            #data = f.read()
            #f.close()
            filepath = sirs_root + toks[0] + ".feats"
            #cl = 0
        elif toks[2] == "4": # septic shock
            #f = open(sepshock_root + toks[0] + ".feats", 'r')
            #data = f.read()
            #f.close()
            filepath = sepshock_root + toks[0] + ".feats"
            #cl = 1
        elif toks[2] == "3": # severe sepsis
            #f = open(sevsep_root + toks[0] + ".feats", 'r')
            #data = f.read()
            filepath = sevsep_root + toks[0] + ".feats"
            #f.close()
        else:
            print "Unidentified class"
        if filepath!= '':
            dt = np.loadtxt(filepath)
            mdt = ma.masked_invalid(dt)
            means = np.array(means)
            mdt = mdt - means
            mdt = np.square(mdt)
            sum = np.sum(mdt, axis=0)
            cumilative_squared_diff = cumilative_squared_diff + sum
    '''
    for line in f:
      toks = line.split()
      if len(toks) < 3:
        continue
      data = None
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
      '''
    variances = cumilative_squared_diff / float(count)
    return list(variances)


def main():
    means, count = get_mean()
    print "count = " + str(count)
    print "means = " + str(means)
    variances = get_variance(means, count)

    print "var = " + str(variances)


if __name__ == "__main__":
    main()
