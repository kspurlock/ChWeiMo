# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:59:03 2021

@author: kylei
"""


import numpy as np
from sklearn import mixture
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class GMMDensity():
    def __init__(self):
        self.gmms = [] #Array for holding class dependent gaussian mixture models
        self.med_densities = [] #Array for holding class dependent densities
        self.params = [] #holds cv_1 and b for each component so they don't need to be recalculated

    def fit_class_gmm(self, X, Y):
        n_classes = len(np.unique(Y))
        
        '''For each class need a seperate GMM model'''
        for i in range(n_classes):
            component_space = X[np.where(Y==i)[0]]
            
            gmm = mixture.GaussianMixture(n_components=1, covariance_type = 'diag')
            
            gmm.fit(component_space)
            self.gmms.append(gmm)
    
    def estimate_class_density(self, X, Y):
        '''Estimate the density for each class,
            using the GMM fitted to each of those classes'''
            
        for j in range(len(self.gmms)):
            densities = []
            
            component_space = X[np.where(Y==j)[0]]
            component_space = component_space+0.00001*np.random.rand(component_space.shape[0], component_space.shape[1])
            
            cov_1 = np.linalg.inv(np.cov(component_space, rowvar = False))
            x_i = self.gmms[j].means_.reshape(-1,1)
            w_i = self.gmms[j].weights_
            dim = X.shape[1]
            
            b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov_1))
            
            self.params.append([cov_1, b])
            for i in range(component_space.shape[0]):
                density = np.dot(component_space[i] - x_i,
                                 np.dot(cov_1, component_space[i] - x_i)) + b
                
                densities.append(density)
            
            #med_density = -2.*np.log(np.median(densities))
            med_density = np.median(densities)
            self.med_densities.append(med_density)
            
    
    def find_new_density(self, x_prime):
        densities = []
        
        for j in range(len(self.gmms)):
            x_i = self.gmms[j].means_.reshape(-1,1)
            cov_1 = self.params[j][0]
            b = self.params[j][1]
            

            density = np.dot(x_prime.reshape(1,-1) - x_i,
                                 np.dot(cov_1, x_prime.reshape(1,-1) - x_i)) + b
            
            densities.append(np.median(density))
        
        return (max(densities))



def preprocessing():
    dataset = pd.read_csv('prototype cf/german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

'''
if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    n_classes = 2
    data, labels = make_blobs(n_samples=1000,
                          centers=n_classes,
                          random_state=10,
                          center_box=(-3.,8.),
                          cluster_std=2.5)

    gmm = GMMDensity()
    gmm.fit_class_gmm(data, labels)

    gmm.estimate_class_density(data, labels)
    
    np.random.seed(19202)
    i = np.random.randint(0, len(data))
    random_sample = data[i]
    label = labels[i]
    
    test = gmm.find_new_density(random_sample)


    print(test < gmm.med_densities[0])
    print(test < gmm.med_densities[1])

    print(gmm.med_densities)

#%%
    dataset = preprocessing()
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,[-1]].values

    gmm = GMMDensity()
    gmm.fit_class_gmm(X, Y)
    gmm.estimate_class_density(X, Y)
'''















