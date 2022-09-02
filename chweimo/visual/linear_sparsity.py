import numpy as np 
from math import log2

def gini(estimator, x, y):
    """
    Grid search target for optimizing GINI as a sparsity measure

    Parameters
    ----------
    estimator : object
        Any sci-kit: learn model capable of interfacing with grid search
    x : np.ndarray
        Explanatory variables
    y : np.ndarray
        Target variable(s)

    Returns
    -------
    total : float64
        GINI ratio of sparsity dispersion

    """
    vector = abs(estimator.coef_)
    
    vec = np.sort(vector)
    N = len(vec)
    l1 = np.sum(np.absolute(vec))
    
    summation = 0
    for i in range(N):
        b = abs(vec[i])/l1
        c = (N - (i+1) + .5)/N
        
        summation += np.sum(b*c)
        
    total = 1 - 2*summation
    
    return total

def kl_divergence(estimator, x, y):
    """
    Grid search target for optimizing GINI as a sparsity measure

    Parameters
    ----------
    estimator : object
        Any sci-kit: learn model capable of interfacing with grid search
    x : np.ndarray
        Explanatory variables
    y : np.ndarray
        Target variable(s)

    Returns
    -------
    float64
        Returns the KL divergence of coeffecient distributions as bits

    """
    coef = abs(estimator.coef_)
    
    p = np.empty((coef.shape[0]), dtype="float64")
    sum_P = sum(coef)
    q = np.empty((coef.shape[0]), dtype="float64")
    
    for i in range(len(coef)):
        p[i] = (coef[i]/sum_P)+1e-2
        
    sum_Q = sum(p)
    
    for i in range(len(coef)):
        q[i] = (p[i]**2/sum_Q**2)+1e-2
        
    a = np.empty((coef.shape[0]), dtype="float64")
    for i in range(len(coef)):
        a[i] = p[i] * log2(p[i]/q[i])
        
    return -sum(a) #bits