# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 23:56:27 2021

@author: Kyle
"""

import numpy as np
from math import log2
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def kl_divergence(estimator, x, y):
    global coef, p, q
    coef = estimator.coef_
    p = np.empty((coef.shape[0],))
    sum_P = sum(coef)
    q = np.empty((coef.shape[0],))
    
    for i in range(len(coef)):
        p[i] = (coef[i]/sum_P)+1e-2
        
    sum_Q = sum(p)
    
    for i in range(len(coef)):
        q[i] = (p[i]**2/sum_Q**2)+1e-2
        
    global a
    a = np.empty((coef.shape[0],))
    for i in range(len(coef)):
        a[i] = p[i] * log2(p[i]/q[i])
        
    return -sum(a) #bits

data = load_diabetes()
x = data.data
y = data.target.reshape(-1,1)

param_grid = {'alpha':np.linspace(.2,10,1)}

clf = GridSearchCV(Lasso(), param_grid,
                   scoring = kl_divergence)

clf.fit(x, y)

best = clf.best_estimator_.coef_




