#
# Python program utilizing the attributes of a JSON file
# to process a dimensionality reduction algorithm, namely
# ghapca
#
# Author: christophe.cerin@inria.fr
# Date  : January 16, 2025
#

# importing the modules
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from ghapca import ghapca

#
# Scalers 
#

def normalize(values):
    return (values - values.min())/(values.max() - values.min())
#
# cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Normalize the feature columns
#
#df[cols] = df[cols].apply(normalize)

def standardize(values):
    return (values - values.mean())/values.std()

#cols = ['hsc_p', 'ssc_p', 'age', 'height', 'salary']
#
# Standardize the feature columns; Dataframe needs to be recreated for
# the following command to work properly.
#
#df[cols] = df[cols].apply(standardize)

# Opening the JSON file
df = pd.read_json('ems-tourperret.ndjson', lines=True)
Nrows = int(df.shape[0])
#print('Nrows:',Nrows)

# Use Itertuples to iterate over payload keys
data = []
for i, row in enumerate(df.itertuples(index=False)):

    val = row[df.columns.get_loc('decoded')]

    if 'payload' in val:
        res = []
        for key in val['payload']:
            #print('\t',key,':',val['payload'][key])
            res.append(val['payload'][key])

        data.append(res)
    else:
        Nrows = Nrows - 1
        print('No payload',val,Nrows)
        
#
# Pre-process the data
#

data = np.array(data)
#
# normalize data
#
data = normalize(data)
        
#print(data)
n = Nrows  # Number of observations
p = 10    # Number of variables

# Initialize parameters for the GHA algorithm
q = 2  # Number of principal components to find
gamma = np.full(q, 1 / n)  # Learning rate (gain parameter)

# Initialize eigenvalues and eigenvectors
lambda_values = np.zeros(q)            # Initial eigenvalues
U = np.random.randn(p, q)              # Initial eigenvectors (random initialization)
U = U / np.sqrt(np.sum(U**2, axis=0))  # Normalize eigenvectors

# Centering vector (mean of each column)
center = np.mean(data, axis=0)

# Apply the GHA algorithm iteratively to each data point
for ii in range(n):
       x = data[ii, :]
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
print('Type of scores:',type(scores))

# Plot the first two principal components
plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("GHA - First Two Principal Components")
plt.show()
        
