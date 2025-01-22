import random
import math
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from ghapca import ghapca
import seaborn as sns
#
# for Kernel density estimation:
#
from sklearn.neighbors import KernelDensity

#
# Utilitty functions
#

def print_arr(a, n):
    for i in range(n):
        print(f"{i} -> {a[i]._x}, {a[i]._y}")
    print()

def is_sorted(a, n):
    for i in range(n-1):
        #print(f"{i} -> {a[i]._x}, {a[i]._y}")
        if greater_points(a[i],a[i+1]):
            return False
    return True

def make_array(my_points):
    a=[]
    b=[]
    for p in my_points:
        a.append(p._x)
        b.append(p._y)
    return np.array(a), np.array(b)

#
# train_test_split from https://www.kaggle.com/code/marwanahmed1911/train-test-split-function-from-scratch
#
def shuffle_data(X, y):

    Data_num = np.arange(X.shape[0])
    np.random.shuffle(Data_num)

    return X[Data_num], y[Data_num]

def train_test_split_scratch(X, y, test_size=0.5, shuffle=True):
    if shuffle:
        X, y = shuffle_data(X, y)
    if test_size <1 :
        train_ratio = len(y) - int(len(y) *test_size)
        X_train, X_test = X[:train_ratio], X[train_ratio:]
        y_train, y_test = y[:train_ratio], y[train_ratio:]
        return X_train, X_test, y_train, y_test
    elif test_size in range(1,len(y)):
        X_train, X_test = X[test_size:], X[:test_size]
        y_train, y_test = y[test_size:], y[:test_size]
        return X_train, X_test, y_train, y_test

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


    
#
# Performance metrics
# MSE: Measures the average squared difference between actual and
# predicted values. Lower is better.
# RMSE: A more interpretable version of MSE (it’s in the same units as
# the target variable).
# MAE: The average absolute difference between the predicted and
# actual values.
# R-squared: Indicates how well the model explains the variance in the
# data. A value close to 1 means the model is doing well.
#
# TODO
#

#
# Main is comming!
#

if __name__ == "__main__":

    W      = 1024
    sqrt_W = int(math.sqrt(W))

    #
    # Buffer points receives random data points and
    # Buffer my_points receives the first block of data 
    #
    # Opening the JSON file
    df = pd.read_json('ems-tourperret.ndjson', lines=True)
    Nrows = int(df.shape[0])
    #print('Nrows:',Nrows)
    No_payload = 0
    
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
            No_payload = No_payload + 1
            #print('No payload',val,Nrows)

    data = np.array(data)
    #
    # Normalize data
    # Attention: we cannot do this in the context of continual
    # learning since we do not know all data when starting!
    #
    data = normalize(data)
    # make a copy
    data_orig = data
    
    #print(data)
    n = W  # Number of observations
    p = 10    # Number of variables

    # Initialize parameters for the GHA algorithm
    q = 2  # Number of principal components to find
    gamma = np.full(q, 1 / n)  # Learning rate (gain parameter)

    # Initialize eigenvalues and eigenvectors
    lambda_values = np.zeros(q)            # Initial eigenvalues
    # Make a copy
    lambda_values_orig = lambda_values
    
    U = np.random.randn(p, q)              # Initial eigenvectors (random initialization)
    U = U / np.sqrt(np.sum(U**2, axis=0))  # Normalize eigenvectors
    # make a copy
    U_orig = U
    # Centering vector (mean of each column)
    center = np.mean(data, axis=0)
    # make a copy
    center_orig = center
    #
    # Work with the first chunck of data
    #
    my_points = [[0,0,0,0,0,0,0,0,0,0] for _ in range(W)]
    for i in range(Nrows):
        if i < W:
            my_points[i] = data[i]
        else:
            break

    my_points_orig = [[0,0,0,0,0,0,0,0,0,0] for _ in range(Nrows)]
    for i in range(Nrows):
            my_points_orig[i] = data_orig[i]
    
    # Apply the GHA algorithm iteratively to each data point
    for ii in range(Nrows):
        x = my_points_orig[ii]
        gha_result_orig = ghapca(lambda_values_orig, U_orig, x, gamma, q, center_orig, sort=True)
        lambda_values_orig = gha_result_orig['values']
        U_orig = gha_result_orig['vectors']
    # final scores for the original ghapca algorithm
    scores_orig = np.dot(my_points_orig, U_orig)

    #print(len(my_points),W,Nrows,No_payload,int(Nrows/W))
    #
    # We sort according to accMotion, pulseAbs, temperature
    # atrtibutes
    #
    my_points = sorted(my_points, key=lambda x: (x[0], x[3], x[4]))    
    my_points = np.array(my_points)

    #
    # We prepare indexes for the regular sampling technique
    #
    mes_indices = [i for i in range(sqrt_W - 1, W - sqrt_W, sqrt_W)]
    #print('Size:',len(mes_indices),'Indexes:',mes_indices)

    # First round
    # Apply the GHA algorithm iteratively to each data point
    for ii in range(n):
        x = my_points[ii, :]
        gha_result = ghapca(lambda_values, U, x, gamma, q, center, sort=True)
        lambda_values = gha_result['values']
        U = gha_result['vectors']

    # Print the results
    #print("Updated Eigenvalues:")
    #print(lambda_values)

    #print("Updated Eigenvectors:")
    #print(U)

    # Project data onto the new principal components
    scores = np.dot(my_points, U)

    #
    # Second, third, etc rounds
    #
    N = int( (Nrows - W) / len(mes_indices))
    #print('N:',N,'Nrows:',Nrows,'W:',W,'mes_indices:',len(mes_indices))
    for i in range(N-1):
        #
        # First we replace some data by a regular sampling technique
        #
        #print(i,W + len(mes_indices)*i)
        K = 0
        for j in mes_indices:
            my_points[j] = data[W + len(mes_indices)*i + K]
            K = K + 1

        # We sort according to accMotion, pulseAbs, temperature
        # atrtibutes
        #
        my_points = sorted(my_points, key=lambda x: (x[0], x[3], x[4]))    
        my_points = np.array(my_points)

        #
        # Now we start the decicated calculation: ghapca
        #
        #for ii in range(n):
        for ii in mes_indices:
            x = my_points[ii, :]
            gha_result = ghapca(lambda_values, U, x, gamma, q, center, sort=True)
            lambda_values = gha_result['values']
            U = gha_result['vectors']

        # Print the results
        #print("Updated Eigenvalues:")
        #print(lambda_values)

        #print("Updated Eigenvectors:")
        #print(U)

        # Project data onto the new principal components
        scores = np.dot(my_points, U)
        #print('------')
        #print(scores[:, 0])
        #print('------')
        #print(scores[:, 1])

    print('----------------------------------------------')
    print('--- Memory footprint extreme edge-inc alg. ---')
    print('Scores dimensions:',scores.shape)
    print('U dimensions:',U.shape)
    print('Lambda_values dimensions:',lambda_values.shape)
    print('Center dimensions:',center.shape)
    print('----------------------------------------------')
    print('--- Memory footprint original ghapca alg.  ---')
    print('Scores dimensions:',scores_orig.shape)
    print('U dimensions:',U_orig.shape)
    print('Lambda_values dimensions:',lambda_values_orig.shape)
    print('Center dimensions:',center_orig.shape)
    print('----------------------------------------------')

    #
    # Kernel density estimation with scikit-learn
    # See: https://scikit-learn.org/1.5/modules/density.html#kernel-density-estimation
    #
    kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(scores)
    my_kde = kde.score_samples(scores)
    print('Kernel Density Estimation (KDE) vector:',my_kde)
    print('--- Statistics on the KDE vector (incremental ghapca) ---')
    print('Max:',np.max(my_kde),'Min:',np.min(my_kde),'Mean:',np.mean(my_kde),'StDev:',np.std(my_kde))

    kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(scores)
    my_kde = kde.score_samples(scores_orig)
    print('Kernel Density Estimation (KDE) vector:',my_kde)
    print('--- Statistics on the KDE vector (full dataset) ---')
    print('Max:',np.max(my_kde),'Min:',np.min(my_kde),'Mean:',np.mean(my_kde),'StDev:',np.std(my_kde))

    #print(len(scores[:, 0]), len(scores[:, 1]))
    # Plot the first two principal components
    plt.scatter(scores[:, 0], scores[:, 1])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("GHA - First Two Principal Components")

    # Kernel density plot
    fig, ax = plt.subplots(1)
    plt.title("kde plot")
    sns.kdeplot(x=scores[:, 0], y=scores[:, 1], fill=True)

    plt.show()
