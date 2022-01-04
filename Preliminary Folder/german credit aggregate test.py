# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:19:42 2021

@author: kylei
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (confusion_matrix,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             plot_confusion_matrix)
from sklearn.preprocessing import LabelEncoder

from Explainer import Explainer
import os

os.environ["OMP_NUM_THREADS"]="1"

def split_by_cm(x, y, model):
    true_neg = [] #C 0,0
    false_neg = [] #C 1, 0
    true_pos = [] #C 1, 1
    false_pos = [] #C 0, 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0, stratify = y)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.rcParams['font.size'] = '20'
    fig = plot_confusion_matrix(model, x_test, y_test, cmap = 'Blues',colorbar = False)

    for i in range(x_test.shape[0]):
        #Four cases
        if y_pred[i] == y_test[i] and y_test[i] == 1:
            true_pos.append(i)
        elif y_pred[i] == y_test[i] and y_test[i] == 0:
            true_neg.append(i)
        elif y_pred[i] != y_test[i] and y_pred[i] == 1:
            false_pos.append(i)
        elif y_pred[i] != y_test[i] and y_pred[i] == 0:
            false_neg.append(i)
        else:
            print('missing case')


    data = np.array([true_neg, false_neg, true_pos, false_pos], dtype = object)
    split_samples = []

    for lst in data:
        split_samples.append(x_test[lst])

    return split_samples, x_train, x_test, y_train, y_test

def preprocessing():
    dataset = pd.read_csv('german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

def train_model(x, y, model):
        cv = KFold(n_splits = 3, random_state = None)
        global total_cm
        total_cm = np.zeros((2,2))
        metric_dict = {}
        split_dict = {}
        
        it = 0
        for train_ind, test_ind in cv.split(x):
            x_train, x_test = x[train_ind], x[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]
            
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            total_cm += confusion_matrix(y_test, y_pred)
            
            metrics = [accuracy_score(y_test, y_pred) *100, 
                       precision_score(y_test, y_pred)*100,
                       recall_score(y_test, y_pred)*100]
            
            metrics = np.round(metrics, decimals = 2)
            
            metric_dict[it] = metrics
            split_dict[it] = (train_ind, test_ind)
            it += 1
            
        return total_cm, metric_dict, split_dict

def find_samples(x_test, model, borderline):
    '''Borderline is a bool var that determines whether to find confident or 
        borderline samples based on predictions'''

    predictions = model.predict_proba(x_test)
    indices = []

    if borderline:
        it = 0
        for i in predictions:
            if i[0] <= .55 and i[0] >= .50 and it not in indices:
                indices.append(it)
            elif i[1] <= .50 and i[1] >= .50 and it not in indices:
                indices.append(it)
            else:
                pass

            it+=1

    if not borderline:
        it = 0
        for i in predictions:
            if i[0] >= .80 and it not in indices:
                indices.append(it)
            elif i[1] >= .80 and it not in indices:
                indices.append(it)
            else:
                pass
    
            it+=1

    sample_pool = x_test[indices]
    x_orig = copy.deepcopy(sample_pool[0])
    x_orig_y = copy.deepcopy(predictions[indices[0]])

    change_class = np.argmin(predictions[indices[0]]) #Change to opposite class

    return x_orig, x_orig_y, change_class


'''Begin main execution'''
if __name__ == '__main__':
    global input_shape

    dataset = preprocessing()

    model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    #model2 = LogisticRegression(max_iter = 1000)

    cols = dataset.iloc[:,:-1].columns.values
    class_names = ['Bad Risk', 'No Risk']
    X = dataset.drop(dataset.columns[-1], axis = 1).values
    Y = dataset.iloc[:, [-1]].values.reshape(-1,)
    
    cm, metrics, splits = train_model(X, Y, model)
    #train_model(X, Y, model2)
    
    x_train, x_test = X[splits[0][0]],  X[splits[0][1]]
    y_train, y_test = Y[splits[0][0]],  Y[splits[0][1]]
 
    cm_splits, x_train, x_test, y_train, y_test = split_by_cm(X, Y, model)
    cm_labels = ['true_neg', 'false_neg', 'true_pos', 'false_pos']
    
    data_maximums = np.max(dataset.iloc[:,:-1])
    
    discrete_map = np.where(data_maximums < 20, 1, 0) # Can use np.where(discrete_map == 1, cols, 0)
    continuous_map = np.where(data_maximums > 20, 1, 0)
    
    
#%%
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from math import log2
    from sklearn.model_selection import GridSearchCV
    
    def gini(estimator, x, y):
        vector = abs(estimator.coef_)
        
        vec = np.sort(vector)
        N = len(vec)
        l1 = 0
        
        for i in vec:
            l1 += abs(i)
        
        summation = 0
        for i in range(N):
            b = abs(vec[i])/l1
            c = (N - (i+1) + .5)/N
            
            summation += sum(b*c)
            
        total = 1 - 2*summation
        
        return total
    
    def kl_divergence(estimator, x, y):
        coef = abs(estimator.coef_)
        
        p = np.empty((coef.shape[0],))
        sum_P = sum(coef)
        q = np.empty((coef.shape[0],))
        
        for i in range(len(coef)):
            p[i] = (coef[i]/sum_P)+1e-2
            
        sum_Q = sum(p)
        
        for i in range(len(coef)):
            q[i] = (p[i]**2/sum_Q**2)+1e-2
            
        a = np.empty((coef.shape[0],))
        for i in range(len(coef)):
            a[i] = p[i] * log2(p[i]/q[i])
            
        return -sum(a) #bits

    def aggregate_delta_percentages(section_X, section_F, section_X_orig,
                                    cm_ind):
        # Shorten column names
        for i in range(len(cols)):
            cols[i] = str(cols[i])[0:5]
        
        # Initialize empty arrays for changes and percentages
        obj1_percentages = np.empty((1,section_X_orig[0].shape[0]))
        obj2_percentages = np.empty((1,section_X_orig[0].shape[0]))
        
        obj1_changes = np.empty((1,section_X_orig[0].shape[0]))
        obj2_changes = np.empty((1,section_X_orig[0].shape[0]))
        for i in range(len(section_X_orig)):
            original_x = section_X_orig[i] # Get the original sample for each
            
            x_subsection = section_X[i] # Get final GA population
            f_subsection = section_F[i] # Get final GA fitness
            
            obj_2 = f_subsection[:,[1]]*-1*100 # Convert fitness back to normal
            obj_1 = (f_subsection[:,[0]]*-1)
            
            # Find the individuals with the best fitness
            obj1_best = x_subsection[np.where(obj_1 == max(obj_1))[0]]
            obj2_best = x_subsection[np.where(obj_2 == max(obj_2))[0]]
            
            # Calculate deltas
            obj1_deltas = obj1_best - original_x
            obj2_deltas = obj2_best - original_x
            
            # Add deltas to arrays
            obj1_changes = np.vstack((obj1_changes, obj1_deltas))
            obj2_changes = np.vstack((obj2_changes, obj2_deltas))
            
            # Find percentage change
            obj1_percent = abs(((obj1_deltas+1) / (original_x+1)))*100
            obj2_percent = abs(((obj2_deltas+1) / (original_x+1)))*100
            
            # Add percentage change to arrays
            obj1_percentages = np.vstack((obj1_percentages, obj1_percent))
            obj2_percentages = np.vstack((obj2_percentages, obj2_percent))
            
        # Deleting the initialized value in the arrays
        obj1_percentages = np.delete(obj1_percentages, 0, axis = 0)
        obj2_percentages = np.delete(obj2_percentages, 0, axis = 0)
        
        obj1_changes = np.delete(obj1_changes, 0, axis = 0)
        obj2_changes = np.delete(obj2_changes, 0, axis = 0)
        
        # Median percentages
        obj1_median_P = np.round(np.median(obj1_percentages, axis = 0),2)
        obj2_median_P = np.round(np.median(obj2_percentages, axis = 0),2)
        
        # Median changes
        obj1_median_C = np.round(np.median(obj1_changes, axis = 0),2)
        obj2_median_C = np.round(np.median(obj2_changes, axis = 0),2)
        
        #######################################################################
        
        # First the change graph (discrete)
        width = 0.35
        discrete_cols = cols[np.where(discrete_map==1)]
        x_discrete = np.arange(0, len(discrete_cols))
        discrete_obj1 = obj1_median_C[np.where(discrete_map==1)]
        discrete_obj2 = obj2_median_C[np.where(discrete_map==1)]
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x_discrete - width/2, discrete_obj1, width,
                        label = 'Obj1', color = 'tab:orange')
        rects2 = ax.bar(x_discrete + width/2, discrete_obj2, width,
                        label = 'Obj2', color = 'tab:blue')
        
        ax.set_ylabel('Change', fontdict={'fontsize':12})
        ax.set_xlabel('Feature', fontdict={'fontsize':12})
        ax.set_title('Median Deltas for Discrete Variables - {s}'.format(s=cm_labels[cm_ind]),
                     fontdict={'fontsize':12})
        ax.set_xticks(x_discrete)
        ax.set_xticklabels(discrete_cols, fontdict={'fontsize':8})
        ax.legend(loc = "upper left", fontsize = 8)
        ax.axhline(linestyle = '--', color = 'black')
        
        ax.bar_label(rects1, padding = 3, fontsize = 8)
        ax.bar_label(rects2, padding = 4, fontsize = 8)
        
        fig.tight_layout()
        plt.show()
        fig.savefig("discrete change - {c}.png".format(c=cm_labels[cm_ind]))
        
        
        # Now the percentage graph (continuous)
        width = 0.35
        cont_cols = cols[np.where(discrete_map==0)]
        x_cont = np.arange(0, len(cont_cols))
        cont_obj1 = obj1_median_P[np.where(discrete_map==0)]
        cont_obj2 = obj2_median_P[np.where(discrete_map==0)]
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x_cont - width/2, cont_obj1, width,
                        label = 'Obj1', color = 'tab:orange')
        rects2 = ax.bar(x_cont + width/2, cont_obj1, width,
                        label = 'Obj2', color = 'tab:blue')
        
        ax.set_ylabel('Change', fontdict={'fontsize':12})
        ax.set_xlabel('Feature', fontdict={'fontsize':12})
        ax.set_title('Median %Change for Continuous Variables - {s}'.format(s=cm_labels[cm_ind]),
                     fontdict={'fontsize':12})
        ax.set_xticks(x_cont)
        ax.set_xticklabels(cont_cols, fontdict={'fontsize':8})
        ax.legend(loc = "upper left", fontsize = 8)
        
        ax.bar_label(rects1, padding = 3, fontsize = 8)
        ax.bar_label(rects2, padding = 4, fontsize = 8)
        
        fig.tight_layout()
        plt.show()
        fig.savefig("cont change - {c}.png".format(c=cm_labels[cm_ind]))
        
        """
        width = 0.35
        x_ = np.arange(0, len(obj1_median))
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x_ - width/2, obj1_median, width,
                        label = 'Obj1', color = 'tab:orange')
        rects2 = ax.bar(x_ + width/2, obj2_median, width,
                        label = 'Obj2', color = 'tab:blue')
        
        ax.set_ylabel('Change', fontdict={'fontsize':12})
        ax.set_xlabel('Feature', fontdict={'fontsize':12})
        ax.set_title('Change Percentages - {s}'.format(s=cm_labels[cm_ind])
                     ,fontdict={'fontsize':12})
        ax.set_xticks(x_)
        ax.set_xticklabels(short_cols, fontdict={'fontsize':8})
        ax.legend(loc = "upper left", fontsize = 8)
        
        ax.bar_label(rects1, padding = 3, fontsize = 8)
        ax.bar_label(rects2, padding = 12, fontsize = 8)
        
        fig.tight_layout()
        plt.show()
        fig.savefig("change - {c}.png".format(c=cm_labels[cm_ind]))
        """
        
    def aggregate_coeff(section_X, section_F, section_X_orig, cm_ind):
        short_cols = cols
        for i in range(len(cols)):
            short_cols[i] = str(cols[i])[0:5]
        
        coef_list_KL = np.empty((1, section_X_orig[0].shape[0]))
        coef_list_Normal = np.empty((1, section_X_orig[0].shape[0]))
        coef_list_GINI = np.empty((1, section_X_orig[0].shape[0]))
        for i in range(len(section_X_orig)):
            original_x = section_X_orig[i]
            
            x_subsection = section_X[i]
            f_subsection = section_F[i]
            
            obj_2 = f_subsection[:,[1]]*-1*100 # Converting obj2 back to normal
            
            deltas = []
            
            for x_new in x_subsection:
                deltas.append(x_new-original_x)
            
            deltas = StandardScaler().fit_transform(np.array(deltas))
            
            # Finding coefficients with KL Grid Search
            param_grid = {'alpha':np.linspace(.1, 2, 20)}
            clf = GridSearchCV(Lasso(), param_grid,
                               scoring = kl_divergence)
            
            clf.fit(deltas, obj_2)
            
            kl_divergence_score = -1*np.round(clf.best_score_,2)
            print(kl_divergence_score)
            coef = abs(clf.best_estimator_.coef_)
            coef_list_KL = np.vstack((coef_list_KL, coef))
            
            # Finding normal coefficients with default LASSO
            regressor = Lasso(alpha = 1) #Intialize
            regressor.fit(deltas, obj_2) #Fit
            coef = abs(regressor.coef_.reshape(1,-1)) #Coefficients
            coef_list_Normal = np.vstack((coef_list_Normal, coef))
            
            
        # Finding median of coefficients for KL grid search
        coef_list_KL = np.delete(coef_list_KL, 0, axis = 0)
        coef_median_KL = np.median(coef_list_KL, axis = 0)
        
        # Finding median of coefficients for GINI grid search
        coef_list_GINI = np.delete(coef_list_GINI, 0, axis = 0)
        coef_median_GINI = np.median(coef_list_GINI, axis = 0)
        
        # Finding median of coefficients WITHOUT grid search
        coef_list_Normal = np.delete(coef_list_Normal, 0, axis = 0)
        coef_median_Normal = np.median(coef_list_Normal, axis = 0)
        
        """
        # GINI for KL grid search coefficients and Normal coefficients
        sparsity_measure_KL = np.round(gini(coef_median_KL.reshape(-1,1)),2)
        sparsity_measure_Normal = np.round(gini(coef_median_Normal.reshape(-1,1)),2)
        """
        
        
        #######################################################################
        width = 0.35
        x_ = np.arange(0, len(coef_median_KL))
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x_ - width/2, np.round(coef_median_Normal,2), width,
                        label = "No Grid Search",
                        color = 'tab:purple')
        rects2 = ax.bar(x_ + width/2, np.round(coef_median_KL, 2), width,
                        label = "With KL divergence",
                        color = 'tab:blue')
        
        ax.set_ylabel('Magnitude', fontdict={'fontsize':12})
        ax.set_xlabel('Feature', fontdict={'fontsize':12})
        
        ax.set_title('Feature Weights - {s}'.format(s=cm_labels[cm_ind]),
                     fontdict={'fontsize':12})
        
        ax.set_xticks(x_)
        ax.set_xticklabels(short_cols, fontdict={'fontsize':8})
        ax.legend(loc = "upper left", fontsize = 8)
        
        ax.bar_label(rects1, padding = 3, fontsize = 8)
        ax.bar_label(rects2, padding = 3, fontsize = 8)
        
        fig.tight_layout()
        plt.show()
        fig.savefig("coef - {c}.png".format(c=cm_labels[cm_ind]))
#%%
    start = time.process_time() #Check how long non-plausible takes
    itera = 0
    for section in cm_splits: #lst is each of the cm sections
        section_F = []
        section_X = []
        section_X_orig = []
        section_pred = []
        
        explainer = Explainer(X, Y, model)
        for sample in section: #sample in each of the confusion matrix
            x_orig = sample
            x_orig_y = model.predict_proba(x_orig.reshape(1,-1)).reshape(-1)

            change_class = np.argmin(x_orig_y) #what class do we need to change to
            
            
            res = explainer.explain_instance(sample = x_orig,
                                             change_class = change_class,
                                             plausible = True,
                                             method = 'NSGA2',
                                             method_params = None)

            section_X.append(res.history[-1].pop.get("X"))
            section_F.append(res.history[-1].pop.get("F"))
            section_X_orig.append(sample)
            section_pred.append(change_class)
            
        '''Once outside need to aggregate change weights and coefficients'''
        
        aggregate_delta_percentages(section_X, section_F,
                                    section_X_orig, itera)
        
        aggregate_coeff(section_X, section_F, section_X_orig, itera)
                                                   
        
        itera+=1
        
    
#%% Begin aggregation
        
        
        