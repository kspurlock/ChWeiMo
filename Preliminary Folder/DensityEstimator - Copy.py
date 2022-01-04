# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 22:59:03 2021

@author: kylei
"""

import numpy as np
from sklearn import mixture
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd

class GMMDensity():
    def __init__(self):
        self.gmms = [] #Array for holding class dependent gaussian mixture models
        self.med_densities = [] #Array for holding class dependent densities
        self.params = [] #holds cv_1 and b for each component so they don't need to be recalculated
        self.dim = []
        
    def return_gmms(self):
        return self.gmms

    def fit_class_gmm(self, X, Y):
        n_classes = len(np.unique(Y))
        
        self.dim = X.shape[0]
        
        '''For each class need a seperate GMM model'''
        for i in range(n_classes):
            component_space = X[Y==i]
            
            cv = GridSearchCV(estimator=mixture.GaussianMixture(covariance_type = 'full')
                              ,param_grid = {'n_components':range(1,10)}, cv = 5)
            
            cv.fit(component_space)
            n_components_ = cv.best_params_['n_components']
            
            gmm = mixture.GaussianMixture(n_components = n_components_, 
                                          covariance_type = 'full')
            gmm.fit(component_space)
            
            self.gmms.append(gmm)
            
            
    def estimate_densities_of_training_samples(self, X, Y):
        densities = []
        densities_ex = []
    
        # Compute densities of all samples
        for i in range(X.shape[0]):
            gmm = self.gmms[Y[i]]    # Select the class dependent GMM
            
            x = X[i,:]
            z = []
            dim = x.shape[0]
            for j in range(gmm.weights_.shape[0]):
                x_i = gmm.means_[j]
                w_i = gmm.weights_[j]
                cov = gmm.covariances_[j]
                cov = np.linalg.inv(cov)
    
                b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))
                z.append(np.dot(x - x_i, np.dot(cov, x - x_i)) + b)
            
            densities.append(np.min(z))
            densities_ex.append(z)
    
        return np.array(densities), np.array(densities_ex, dtype = object)
    
    def prepare_computation_of_plausible_counterfactuals(self, X, Y):
        densities, densities_ex = self.estimate_densities_of_training_samples(X, Y)
        
        Y_targets = np.unique(Y)
        
        density_threshold = [np.median(densities[Y==y_]) for y_ in Y_targets]
        
        return density_threshold
        
    
    def estimate_class_density(self, X, Y):
        '''Estimate the density for each class,
            using the GMM fitted to each of those classes'''
        dense = []
        for j in range(len(self.gmms)):
            print(j)
            densities = []
            
            component_space = X[Y == j]
            '''
            component_space = component_space+1e-5*np.random.rand(component_space.shape[0],
                                                                  component_space.shape[1])
            '''
            cov_1 = np.linalg.inv(self.gmms[j].covariances_)
            x_i = self.gmms[j].means_.reshape(-1,1)
            w_i = self.gmms[j].weights_
            dim = X.shape[1]
            
            b = -2.*np.log(w_i) + dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov_1))
            
            self.params.append([cov_1, b])
            for i in range(component_space.shape[0]):
                density = np.dot(component_space[i] - x_i,
                                 np.dot(cov_1, component_space[i] - x_i)) + b
                
                densities.append(density)
            
            med_density = -2.*np.log(np.median(densities))
            dense.append(med_density)
            self.med_densities.append(med_density)
            
        return dense
            
    
    def find_new_density(self, x_prime):
        densities = []
        
        for j in range(len(self.gmms)):
            gmm = self.gmms[j]
            
            x_i = gmm.means_
            w_i = gmm.weights_[j]
            cov = gmm.covariances_[j]
            cov = np.linalg.inv(cov)
            
            b = -2.*np.log(w_i) + self.dim*np.log(2.*np.pi) - np.log(np.linalg.det(cov))

            density = np.dot(x_prime - x_i,
                                 np.dot(cov, x_prime - x_i)) + b
        
        return density



def preprocessing():
    dataset = pd.read_csv('prototype cf/german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    n_classes = 2
    data, labels = make_blobs(n_samples=1000,
                          centers=n_classes,
                          random_state=10,
                          center_box=(-3.,8.),
                          cluster_std=2.5)
    
    gmm_obj = GMMDensity()
    gmm_obj.fit_class_gmm(data,labels)
    
    gmm_obj.estimate_densities_of_training_samples(data, labels)

    test = gmm_obj.prepare_computation_of_plausible_counterfactuals(data, labels)
    #%%
    
    np.random.seed()
    i = np.random.randint(0, len(data))
    sample = data[i,:]
    label = labels[i]
    
    gmm_obj.find_new_density(sample)
    #%%
    
    

    #%%
    dataset = preprocessing()
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,[-1]].values

    gmm = GMMDensity()
    gmm.fit_class_gmm(X, Y)
    gmm.estimate_class_density(X, Y)
















