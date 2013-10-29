import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic

class MLP:
  def __init__(self, n_input, n_hidden, n_output, eta=1):

    # At least for MLP, doesn't seem like we actually need persistent nodes.
    #self.input_nodes = np.zeros(n_input )
    #self.hidden_nodes = np.zeros(n_hidden)
    #self.output_nodes = np.zeros(n_output)
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.n_output = n_output

    weights = []
    for i in range(n_input + 1):
      weights.append(np.random.uniform(-1, 1, size=(n_hidden)))
    self.V = np.array(weights)
    weights = []
    for i in range(n_hidden + 1):
      weights.append(np.random.uniform(-1, 1, size=(n_output)))
    self.W = np.array(weights)
    self.A = None # Result of V * X^T
    self.Y = None # Result of W * A^T
    self.eta = 1
 
  def set_learning_rate(eta):
    self.eta = eta

  def forwards(self, X):
    np.append(X, -1)
    print "X = " + str(X)
    print "V = " + str(self.V)
    A = self.g(self.V.dot(X.T))
    print "A = " + str(A)
    print "W = " + str(self.W)
    Y = self.g(self.W.T.dot(A.T))
    self.A = A
    self.Y = Y
    print "Y = " + str(Y)

  def backwards(self, T, X):
    d_o = T - self.Y
    for i in range(0, len(d_o)):
      d_o[i] *= self.Y[i] * (1 - self.Y[i])
    print "d_o = " + str(d_o)
    d_h = np.array(self.A, copy=True)
    for i in range(0, len(d_h)):
      v = 0
      for j in range(0, len(d_o)):
        v += self.W[i][j] * d_o[j]
      d_h[i] *= (1 - self.A[i]) * v
    print "d_h = " + str(d_h)
    
    for i in range(0, len(self.W)):
      for j in range(0, len(self.W[i])):
        self.W[i][j] += self.eta * d_o[j] * self.A[i]

    for i in range(0, len(self.V)):
      for j in range(0, len(self.V[i])):
        self.V[i][j] += self.eta * d_h[i] * X[j]
    
    print "V = " + str(self.V)
    print "W = " + str(self.W)

  def g(self, A):
    return logistic.pdf(A)
    


def main():
  mlp = MLP(2, 2, 2)
  X = np.array([1, 0])
  mlp.forwards(X)
  T = np.array([1, 0])
  mlp.backwards(T, X)


if __name__ == "__main__":
  main()

