import numpy as np
import sys
import math

root = "../../../"
data_path = root + "numerics/"
sirs_root = data_path
sepshock_root = data_path
sevsep_root = data_path
train_map = "../../balanced_trainset.recs"

def handle_cols(cols):
  cols = cols.replace("Time", "Time\t")
  cols = cols.split("\t")
  cols = [x.replace(" ", "_") for x in cols]
  cols =  " ".join(map(str, cols))
  cols = cols.replace("___NBP NBP_Sys BP_Dias BP_Mean", "nbp nbpsys nbpdias nbpmean")
  cols = cols.replace("NBP_Sys BP_Dias BP_Mean", "nbpsys nbpdias nbpmean")
  cols = cols.replace("___ABP ABP_Sys BP_Dias BP_Mean", "abp abpsys abpdias abpmean")
  cols = cols.replace("ABP_Sys BP_Dias BP_Mean", "abpsys abpdias abpmean")
  cols = cols.replace("___PAP PAP_Sys AP_Dias AP_Mean", "pap papsys papdias pappmean")
  cols = cols.replace("___ART ART_Sys RT_Dias RT_Mean", "art artsys artdias artpmean")
  cols = cols.replace("_NBPSys", "nbpsys")
  cols = cols.replace("NBPSys", "nbpsys")
  cols = cols.replace("NBPMean", "nbpmean")
  cols = cols.replace("NBPDias", "nbpdias")
  cols = cols.replace("_ABPSys", "abpsys")
  cols = cols.replace("ABPSys", "abpsys")
  cols = cols.replace("ABPMean", "abpmean")
  cols = cols.replace("ABPDias", "abpdias")
  cols = cols.replace("___RESP", "resp")
  cols = cols.replace("___SpO2", "o2")
  cols = cols.replace("_____HR", "hr")
  cols = cols.replace("__PULSE", "pulse")
  cols = cols.replace("____CVP", "cvp")
  cols = cols.replace("CO", "co")
 # cols = cols.replace("BLOODT", "bloodt")
 # cols = cols.replace("SpO2	 Minute	 Status Status	c Count", "X")
 # cols = cols.replace("Status	c Count", "field_b field_c")
 # cols = cols.replace("Minute Status", "field_a")
  return cols

def check_cols(data):
  label_cols = data[0].split()
  measure_cols = data[1].split()
  if len(label_cols) != len(measure_cols):
    print "label_cols = ", len(label_cols)
    print "measure_cols = ", len(measure_cols)
    print data[1]
    return False
  return True
  
def get_means():
  counts = {"hr": 0, "resp" : 0, "o2": 0, "nbpsys": 0, "nbpdias": 0, "nbpmean": 0}
  count = 0
  sm = 0
  mx = None
  mn = None
  f = open(train_map, 'r').readlines()
  for line in f:
    count = count + 1
    #if count > 100:
    #  return
    sys.stderr.write(".")
    toks = line.split()
    if len(toks) < 2:
      print "toks < 2"
      continue
    data = None
    print toks[0]
    try:
      if toks[2] == "1": # sirs
        f = open(sirs_root + toks[0] + ".sig", 'r')
        data = f.read()
        f.close()
      elif toks[2] == "4": # septic shock
        f = open(sepshock_root + toks[0] + ".sig", 'r')
        data = f.read()
        f.close()
      elif toks[2] == "3": # severe sepsis
        f = open(sevsep_root + toks[0] + ".sig", 'r')
        data = f.read()
        f.close()
      else:
        sys.stderr.write("Unidentified class")
        print "unID class"
        continue
    except:
      sys.stderr.write("Missing file: " + str(toks))
      continue
    if len(data.split('\n')) < 10:
      print "continuing"
      continue
    lines = data.split("\n")
    print lines[0]
    cols = handle_cols(lines[0])
    print cols
    lines[0] = cols
    if(not check_cols(lines)):
      return
    
  """
    return
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
  """

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
  #mean, mx, mn, count = get_mean()
  get_means()
  #stdev = get_stdev(mean, count)
  #print "count = " + str(count)
  #print "mean = " + str(mean) 
  #print "stdev = " + str(stdev)
  #print "max = " + str(mx)
  #print "min = " + str(mn)


if __name__ == "__main__":
  main()
