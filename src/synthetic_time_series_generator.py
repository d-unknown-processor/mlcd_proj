import random
import sys
def f(corrupt, k, max_n):
  to_corrupt = False
  original_max_n = max_n
  for i in range(0, k):
    if not to_corrupt and corrupt and random.random() > .90:
      to_corrupt = True
    if to_corrupt == True:
      max_n = int(max_n - 1)
    random_minus = int(max_n/4.0)
    minus = 0
    if random_minus > 0:
      minus = random.randint(0, int(max_n/4.0))
    if max_n - minus <= 0:
      n = 1
    else:
      n = random.randint(0, max_n - minus)
    for j in range(0, n):
      if random.random() > .9:
        continue
      print j - random.gauss(0,0.01)
    for j in range(0, n):
      if random.random() > .9:
        continue
      print n-j - random.gauss(0,0.01)

def main():
  corrupt = False
  if sys.argv[1] == "1":
    corrupt = True
  f(corrupt, int(sys.argv[2]), int(sys.argv[3]))
if __name__ == "__main__":
  main()
