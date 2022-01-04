# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 04:10:32 2021

@author: kylei
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from Explainer import Explainer

dataset = pd.read_csv("X_data.csv").iloc[0:10000,:]
#%%
def ExhaustiveGrid(x, y, folds=3):
    '''This function performs KFold like train_model as well as grid search.
            Hopefully this optimizes both the training/test set as well as
            parameters.
    '''
    #lets assume that x and y are x_val and y_val

    param_grid = [{'n_estimators': [100, 1000],  # Parameters to optimize
                   'criterion': ['gini', 'entropy']}]

    # Used to evaluate parameters, recall can also be used but
    scoring = ['precision', 'recall']
    #requires more iterations
    
    best_params = []  # Holds parameters, means, and split indices

    #Starting grid search for metrics
    print()
    print('Best params for: %s_macro' % scoring[0])
    print()

    clf = GridSearchCV(RandomForestClassifier(), param_grid,  # GridSearch obj
                       scoring="%s_macro" % scoring[0])

    clf.fit(x, y.reshape(-1,))  # Fit grid search to train set

    print(clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    # Store information from iteration
    best_params.append([clf.best_params_, max(means)])

    for mean, std, params in zip(means, stds,
                                 clf.cv_results_['params']):
        print("%0.3f (+- %0.03f) for %r"
              % (mean, std * 2, params))
    
    best_params = np.array(best_params)
    # Finding optimal params from max mean score
    best_index = np.argmax(best_params[:, [1]])

    params = best_params[best_index][0]
    return params

if __name__ == "__main__":
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,[-1]].values
    #%%
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.10)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=.5)
    
    #%%
    model = RandomForestClassifier(n_estimators=10, criterion="entropy")
    model.fit(x_train, y_train.reshape(-1,))
    pred = model.predict_proba(x_test)
    #%%
    random_index = np.random.randint(0, len(x_test))
    x_orig = x_test[random_index]
    change_class = np.argmin(pred[random_index])
    x_orig_y = y_test[random_index]
    cols = dataset.columns[:-1].values.astype("str")
    class_names = ['stable grasp', 'unstable grasp']
    
    #%%
    explainer = Explainer(x_test, y_test, model)
    res = explainer.explain_instance(sample = x_orig,
                                     change_class = change_class,
                                     plausible = True,
                                     method = 'NSGA2',
                                     method_params = None)
    
    '''
    explainer2 = Explainer(X, Y, model2)
    res2 = explainer2.explain_instance(sample = x_orig,
                                     change_class = change_class,
                                     plausible = True,
                                     method = 'NSGA2',
                                     method_params = None)
    '''


    #%%
    import streamlit as st
    
    '''Version 2 of change weight display'''
    def find_change_weight(opt_result):
        algorithm = opt_result.history[-1] #Consider only final population
        F_pop = algorithm.pop.get("F") #Get objective values
        F_pop[:,[0]] = (-1)/F_pop[:,[0]]
        F_pop[:,[1]] = F_pop[:,[1]]*-1
        X_pop = algorithm.pop.get('X') #Get pop samples
        
        deltas = []
        for i in range(X_pop.shape[0]):
            deltas.append(X_pop[i] - x_orig)
            
        f_x = np.hstack((deltas, F_pop))
        
        global top5_obj1_df, top5_obj2_df, f_x_df_sliced
        new_cols = []
        for i in range(len(cols)):
            new_cols.append(str(cols[i]))
        
        new_cols = np.append(new_cols, "obj1")
        new_cols = np.append(new_cols, "obj2")
        
        f_x_df = pd.DataFrame(data=f_x, columns = new_cols)
        
        f_x_df_sliced = f_x_df.sort_values(by = ["obj1"], ascending = True).iloc[np.r_[0:5, -10:-5],:]
        
        return f_x_df_sliced

    test = find_change_weight(res)
    test.style #style params are in the jupyter file
    
#%%

    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from math import log2
    
    def gini(vector):
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
        global p, q
        coef = abs(estimator.coef_)
        
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

    def print_delta(x_orig, x_prime_ob1, x_prime_ob2, pred1, pred2, cols):
        deltas1 = x_prime_ob1-x_orig
        percent1 = np.round(deltas1/abs(x_orig)*100,2)

        print('Your original class probability is: {base}'.format(base = x_orig_y))

        print('-'*60)

        print('Minimal change required to reach class {c}'
              .format(c=class_names[change_class]))
        for name, change, per in zip(cols, deltas1, percent1):
            print('Need to change feature {n} by: {c} ({p}%)'
                  .format(c = np.round(change,4), n = name, p=per))
        print('Class probability will be: {p1}'.format(p1 = pred1))
        
        print('-'*60)
        
        print('Change required to reach class {c} with maximum probability'
              .format(c=class_names[change_class]))
        deltas2 = x_prime_ob2-x_orig
        percent2 = np.round(deltas2/abs(x_orig)*100,2)

        for name, change, per in zip(cols, deltas2, percent2):
            print('Need to change feature {n} by: {c}  ({p}%)'
                  .format(c = np.round(change,4), n = name, p = per))
            
        print('Class probability will be: {p2}'.format(p2 = pred2))
        
        return percent1, percent2

    def convert(lst):
        #Used to convert a dict sorted into a list back into a dict
        dct = {lst[i][0]: lst[i][1] for i in range(0, len(lst))}
        return dct

    def find_change_weight(opt_result):
        algorithm = opt_result.history[-1] #Consider only final population
    
        F_pop = algorithm.pop.get("F") #Get objective values
        X_pop = algorithm.pop.get('X') #Get pop samples

        obj_2 = F_pop[:,[1]]*-1*100
        obj_1 = (F_pop[:,[0]]*-1)

        deltas = []
        for i in range(X_pop.shape[0]):
            deltas.append(X_pop[i] - x_orig)

        deltas = StandardScaler().fit_transform(np.array(deltas))

        '''Fit the regressor to delta and obj2 and get coefficients'''
        """
        regressor = Lasso(alpha = 1) #Intialize
        regressor.fit(deltas, obj_2) #Fit
        coef = regressor.coef_.reshape(-1,1) #Coefficients
        """
        param_grid = {'alpha':np.linspace(.1, 2, 40)}
        clf = GridSearchCV(Lasso(), param_grid,
                           scoring = kl_divergence)
        
        clf.fit(deltas, obj_2)
        
        kl_divergence_score = -1*np.round(clf.best_score_,2)
        coef = abs(clf.best_estimator_.coef_)
        coef = coef.reshape(-1,1)
        
        sparsity_measure = gini(coef)

        for i, j in zip(cols, coef): #Output coefficient per feature
            print('{c}: {co}'.format(c = i, co = np.round(j[0],4)))

        best_sample = X_pop[np.where(obj_2 == max(obj_2))[0][0]]
        best_sample_prediction = model.predict_proba(best_sample.reshape(1,-1)).reshape(-1)

        best_sample1 = X_pop[np.where(obj_1 == max(obj_1))[0][0]]
        best_sample1_prediction = model.predict_proba(best_sample1.reshape(1,-1)).reshape(-1)

        short_cols = cols
        for i in range(len(cols)):
            short_cols[i] = str(cols[i])
        dct = {key: None for key in short_cols}

        for i in range(len(dct)):
            dct[short_cols[i]] = coef[i][0]
        #dct = sorted(dct.items(), key=lambda x:abs(x[1]), reverse=False)
        #dct = convert(dct)

        ticks = list(dct.keys())

        plt.barh(np.arange(0,len(dct.keys())), dct.values(), color = 'tab:green')
        plt.yticks(np.arange(0,len(dct.keys())), ticks, fontsize=9)
        plt.title('Counterfactual Feature Weights - Credit Risk - P')
        plt.axvline(x = 0, color = 'black', linestyle = '--')
        plt.ylabel('Coefficient/Feature')
        plt.xlabel('Feature Weight')
        plt.legend([np.round(sparsity_measure,3)], title='GINI')
        
        plt.show()
        
        ######################################################################
        ##                     Graphing Change Weights                      ##
        ######################################################################
        
        p1, p2 = print_delta(x_orig, best_sample1, best_sample,
                             best_sample1_prediction, best_sample_prediction,
                             cols)
        
        for i in range(len(p1)):
            if p1[i] == np.inf:
                p1[i] = 1
            
            if p2[i] == np.inf:
                p2[i] = 1
        
        p1 = np.round(abs(p1), 2)
        p2 = np.round(abs(p2), 2)
        x_ = np.arange(len(short_cols))
        width = 0.35
        
        fig, ax = plt.subplots()
        rects1 = ax.barh(x_ - width/2, p1, width, label = 'Obj1', color = 'tab:orange')
        rects2 = ax.barh(x_ + width/2, p2, width, label = 'Obj2', color = 'tab:blue')
        
        ax.set_xlabel('Change')
        ax.set_ylabel('Feature')
        ax.set_title('Change Percentages')
        ax.set_yticks(x_)
        ax.set_yticklabels(short_cols)
        ax.legend()
        
        ax.bar_label(rects1, padding = 3, fontsize = 7)
        ax.bar_label(rects2, padding = 40, fontsize = 7)
        
        fig.tight_layout()
        
        plt.show()
        
        print('Sparsity: {s}'.format(s = sparsity_measure))

        return coef


    coef1 = find_change_weight(res)
    print('#'*60)
    #coef2 = find_change_weight(res2)
    
    
    