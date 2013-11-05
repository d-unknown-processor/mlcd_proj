import matplotlib.pyplot as plt
def get_data():
  f = open("weights.txt", 'r')
  layer0 = []
  layer1 = []
  layer2 = []
  layer3 = []
  for l in f:
    ws = l.split()
    if ws[0] == "0":
      layer0.append(float(ws[2]))
    if ws[0] == "1":
      layer1.append(float(ws[2]))
    if ws[0] == "2":
      layer2.append(float(ws[2]))
    if ws[0] == "3":
      layer3.append(float(ws[2]))
  return layer0, layer1, layer2, layer3

def main():
  l0, l1, l2, l3 = get_data()
  x = range(0,len(l1))
  plt.plot(x, l0, x, l1, x, l2, x, l3)
  plt.legend(['l0', 'l1', 'l2', 'O'], loc="upper right")
  plt.savefig("weights.pdf")
 

if __name__ == "__main__":
  main()
