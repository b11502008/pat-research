import numpy as np
import matplotlib.pyplot as plt

print("--- Task 1: Linear Regression ---")
# Find Beta

np.random.seed(42)
n = 200     # number of samples
d = 3 # number of features for each sample
X = np.random.rand(n, d) * 10 # input features

true_beta = np.array([[4.5], 
                      [-2.0], 
                      [1.5]])
y_data = X @ true_beta + np.random.randn(n, 1) * 1.5 # ground truth labels (answer)

b = np.random.randn(d, 1) # initial weight guess
s = 0.01 # step size
K = 300 # iteration count

loss_record = [] # trajectory of GD

for k in range(K):
    y = X @ b
    error = y - y_data
    loss = (1 / (2 * n)) * np.sum(error**2)
    loss_record.append(loss)
    if loss != 0 :
        sigma = (1 / n) * (X.T @ error)
        b = b - sigma * s
    else:
        break

print("---Gradient Descend Results---")
print(f"True Beta: { true_beta.T }")
print(f"Best Guess: { b.T }")
loss = np.sum((true_beta - b)**2)
print(f"Loss: { loss }")