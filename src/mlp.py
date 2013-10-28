import numpy as np
from scipy import linalg

class MLP:
  def __init__(self, n_input, n_hidden, n_output):
    self.input_nodes = np.zeros(n_input )
    self.hidden_nodes = np.zeros(n_hidden)
    self.output_nodes = np.zeros(n_output)
    weights = []
    for i in range(len(self.input_nodes)):
      weights.append(np.random.uniform(-1, 1, size=(len(self.hidden_nodes) + 1)))
    self.V = np.array(weights)
    weights = []
    for i in range(len(self.hidden_nodes)):
      weights.append(np.random.uniform(-1, 1, size=(len(self.output_nodes) + 1)))
    self.W = np.array(weights)
    #print "V = " + str(self.V)
    #print "W = " + str(self.W)

  def 
    


def main():
  mlp = MLP(2, 2, 1)


if __name__ == "__main__":
  main()

