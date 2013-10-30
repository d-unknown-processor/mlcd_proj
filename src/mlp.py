import numpy as np
from scipy import linalg as lin
from scipy.stats import logistic
import math

class MLP:
  def __init__(self, n_input, n_hidden, n_output, eta=1):

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
    self.A = None # Result of X * V
    self.Y = None # Result of A * W
    self.eta = eta
 
  def set_learning_rate(eta):
    self.eta = eta

  def forwards(self, X):
    X = np.append(X, 1)  # A bias used by each of the hidden nodes
    A = self.g(X.dot(self.V))
    A = np.append(A, 1) # Bias for each output node
    self.A = A
    Y = self.g(A.dot(self.W))
    self.Y = Y

  def backwards(self, T, X):
    X = np.append(X, 1)  # A bias used by each of the hidden nodes
    
    # Compute error d_o  at output
    d_o = T - self.Y
    for i in range(0, len(d_o)):
      d_o[i] *= self.Y[i] * (1 - self.Y[i])

    # Compute error d_h at hidden
    d_h = np.array(self.A, copy=True)
    for i in range(0, len(d_h)):
      v = 0
      for j in range(0, len(d_o)):
        v += self.W[i][j] * d_o[j]
      d_h[i] *= (1 - self.A[i]) * v
   
    # Update weights from hidden to output
    W = []
    for i in range(0, len(self.W)):
      w = []
      for j in range(0, len(self.W[i])):
        w.append(self.W[i][j] + self.eta * d_o[j] * self.A[i])
      W.append(w)
    self.W = np.array(W) 

    # Update weights from input to hidden
    V = []
    for i in range(0, len(self.V)):
      v = []
      for j in range(0, len(self.V[i])):
        v.append(self.V[i][j] + self.eta * d_h[j] * X[i])
      V.append(v)
    self.V = np.array(V)

  # Here g is a logistic function
  def g(self, A):
    B = []
    for i in range(0, len(A)):
      b = 1.0/(1.0 + math.exp(-A[i]))
      B.append(b)
    return np.array(B)

def main():
  n_input = 2
  n_hidden = 2
  n_output = 1
  mlp = MLP(n_input, n_hidden, n_output)
  n_epochs = 5 # Just for testing that learning happens
  for i in range(0, n_epochs):
    X = np.array([1, 0])
    mlp.forwards(X)
    T = np.array([0])
    mlp.backwards(T, X)

if __name__ == "__main__":
  main()

