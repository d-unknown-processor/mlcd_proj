import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic

class MLP:
  def __init__(self, n_input, n_hidden, n_output):

    # At least for MLP, doesn't seem like we actually need persistent nodes.
    #self.input_nodes = np.zeros(n_input )
    #self.hidden_nodes = np.zeros(n_hidden)
    #self.output_nodes = np.zeros(n_output)
    weights = []
    for i in range(len(self.input_nodes) + 1):
      weights.append(np.random.uniform(-1, 1, size=(len(self.hidden_nodes))))
    self.V = np.array(weights)
    weights = []
    for i in range(len(self.hidden_nodes) + 1):
      weights.append(np.random.uniform(-1, 1, size=(len(self.output_nodes))))
    self.W = np.array(weights)

  def forwards(self, X):
    np.append(X, -1)
    print "X = " + str(X)
    print "V = " + str(self.V)
    A = self.g(self.V.dot(X.T))
    print "A = " + str(A)
    print "W = " + str(self.W)
    O = self.g(A.dot(self.W))
    print "O = " + str(O)

  def backwards(self):

  def g(self, A):
    return logistic.pdf(A)
    


def main():
  mlp = MLP(2, 2, 1)
  X = np.array([1, 0])
  mlp.forwards(X)


if __name__ == "__main__":
  main()

