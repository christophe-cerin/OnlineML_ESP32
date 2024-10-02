import numpy as np


def ghapca_C(Q, x, y, gamma):
    """
    Update the matrix Q based on vectors x, y, and gamma.
    
    Parameters:
    Q : numpy.ndarray
        The matrix to be updated.
    x : numpy.ndarray
        The input vector.
    y : numpy.ndarray
        The projected vector.
    gamma : numpy.ndarray
        The learning rate vector.
    
    Returns:
    numpy.ndarray
        The updated matrix Q.
    """
    n, k = Q.shape
    
    # Update Q based on x, y, and gamma
    for i in range(k):
        for j in range(n):
            Q[j, i] += gamma[i] * (x[j] - Q[j, i] * y[i])
    
    return Q

def ghapca(lambda_values, U, x, gamma, q=None, center=None, sort=True):
    """
    Perform online PCA update.
    
    Parameters:
    lambda_values : numpy.ndarray or None
        The eigenvalues.
    U : numpy.ndarray
        The eigenvectors.
    x : numpy.ndarray
        The new data point.
    gamma : float or numpy.ndarray
        The learning rate.
    q : int or None
        The number of principal components to keep.
    center : numpy.ndarray or None
        The center to subtract from x.
    sort : bool
        Whether to sort the eigenvalues and eigenvectors.
    
    Returns:
    dict
        A dictionary with keys 'values' and 'vectors' for eigenvalues and eigenvectors.
    """
    d = U.shape[0]
    k = U.shape[1]
    
    if len(x) != d:
        raise ValueError("Length of x must be equal to the number of rows in U.")
    
    if lambda_values is not None:
        if len(lambda_values) != k:
            raise ValueError("Length of lambda must be equal to the number of columns in U.")
    
    if center is not None:
        x = x - center
    
    gamma = np.resize(gamma, k)
    
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    
    y = np.dot(U.T, x)
    
    U = ghapca_C(U, x, y, gamma)
    
    if lambda_values is not None:
        lambda_values = (1 - gamma) * lambda_values + gamma * y ** 2
        if sort:
            ix = np.argsort(-lambda_values)  # Sorting in decreasing order
            if not np.array_equal(ix, np.arange(k)):
                lambda_values = lambda_values[ix]
                U = U[:, ix]
        if q is not None and q < k:
            lambda_values = lambda_values[:q]
    else:
        lambda_values = None
    
    if q is not None and q < k:
        U = U[:, :q]
    
    return {'values': lambda_values, 'vectors': U}

