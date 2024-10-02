import numpy as np

import matplotlib.pyplot as plt

from ghapca import ghapca

# Set the random seed for reproducibility
np.random.seed(123)

# Generate some synthetic multivariate data for testing
n = 100  # Number of observations
p = 5    # Number of variables
data = np.random.randn(n, p)
print(data)
# Initialize parameters for the GHA algorithm
q = 2  # Number of principal components to find
gamma = np.full(q, 1 / n)  # Learning rate (gain parameter)

# Initialize eigenvalues and eigenvectors
lambda_values = np.zeros(q)  # Initial eigenvalues
U = np.random.randn(p, q)  # Initial eigenvectors (random initialization)
U = U / np.sqrt(np.sum(U**2, axis=0))  # Normalize eigenvectors

# Centering vector (mean of each column)
center = np.mean(data, axis=0)

# Apply the GHA algorithm iteratively to each data point
for i in range(n):
    x = data[i, :]
    gha_result = ghapca(lambda_values, U, x, gamma, q, center, sort=True)
    lambda_values = gha_result['values']
    U = gha_result['vectors']

# Print the results
print("Updated Eigenvalues:")
print(lambda_values)

print("Updated Eigenvectors:")
print(U)

# Project data onto the new principal components
scores = np.dot(data, U)

# Plot the first two principal components
plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GHA - First Two Principal Components")
plt.show()
