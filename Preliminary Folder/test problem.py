# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 23:19:42 2021

@author: kylei
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


from Explainer import Explainer

def preprocessing():
    dataset = pd.read_csv(r'german_credit_data.csv').drop('Unnamed: 0', axis = 1)
    
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

#%%
'''Begin main execution'''
if __name__ == '__main__':
    global input_shape

    dataset = preprocessing()

    model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    model2 = LogisticRegression(max_iter = 1000)

    cols = dataset.iloc[:,:-1].columns.values
    class_names = ['Bad Risk', 'No Risk']
    X = dataset.drop(dataset.columns[-1], axis = 1).values
    Y = dataset.iloc[:, [-1]].values.reshape(-1,)
    
    cm, metrics, splits = train_model(X, Y, model)
    train_model(X, Y, model2)
    
    x_train, x_test = X[splits[0][0]],  X[splits[0][1]]
    y_train, y_test = Y[splits[0][0]],  Y[splits[0][1]]

    '''Precomputing required density elements'''
 
    x_orig, x_orig_y, change_class = find_samples(x_test, model, False)
    change_class = np.argmin(x_orig_y)

    
    
    #%%
    explainer = Explainer(X, Y, model)
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
        regressor = Lasso(alpha = 1) #Intialize
        regressor.fit(deltas, obj_2) #Fit
        coef = regressor.coef_.reshape(-1,1) #Coefficients
        
        sparsity_measure = gini(coef)

        for i, j in zip(cols, coef): #Output coefficient per feature
            print('{c}: {co}'.format(c = i, co = np.round(j[0],4)))

        best_sample = X_pop[np.where(obj_2 == max(obj_2))[0][0]]
        best_sample_prediction = model.predict_proba(best_sample.reshape(1,-1)).reshape(-1)

        best_sample1 = X_pop[np.where(obj_1 == max(obj_1))[0][0]]
        best_sample1_prediction = model.predict_proba(best_sample1.reshape(1,-1)).reshape(-1)

        short_cols = cols
        for i in range(len(cols)):
            short_cols[i] = str(cols[i])[0:5]
        dct = {key: None for key in short_cols}

        for i in range(len(dct)):
            dct[short_cols[i]] = coef[i][0]

        dct = sorted(dct.items(), key=lambda x:abs(x[1]), reverse=False)
        dct = convert(dct)

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
        rects1 = ax.bar(x_ - width/2, p1, width, label = 'Obj1', color = 'tab:orange')
        rects2 = ax.bar(x_ + width/2, p2, width, label = 'Obj2', color = 'tab:blue')
        
        ax.set_ylabel('Change')
        ax.set_xlabel('Feature')
        ax.set_title('Change Percentages')
        ax.set_xticks(x_)
        ax.set_xticklabels(short_cols)
        ax.legend()
        
        ax.bar_label(rects1, padding = 3, fontsize = 8)
        ax.bar_label(rects2, padding = 12, fontsize = 8)
        
        fig.tight_layout()
        
        plt.show()
        
        print('Sparsity: {s}'.format(s = sparsity_measure))

        return coef


    coef1 = find_change_weight(res)
    print('#'*60)
    #coef2 = find_change_weight(res2)
#%%
    if (1):
        pass
    else:
        #Section allows for real-time graphing of the best objective values'''
        def find_nearest(array, value):
            #'''Search func to find closest value to median objective function'''
            idx = 0
            smallest = 0
    
            for i in range(len(array)):
                val = abs(array[i][0] - value[0]) + abs(array[i][1] - value[1])
    
                if val < smallest or i == 0:
                    smallest = val
                    idx = i
    
            return idx
    
        obj_1_X = []
        obj_2_X = []
    
        mid_X1 = []
        mid_X2 = []
        
        obj_1_F = []
        obj_2_F = []
    
        # iterate over the deepcopies of algorithms
        
        for algorithm in res.history:
            F_pop = algorithm.pop.get("F")
            X_pop = algorithm.pop.get('X')
        
            min_1 = np.where(F_pop[:,[0]] == min(F_pop[:,[0]]))[0][0]
            min_2 = np.where(F_pop[:,[1]] == min(F_pop[:,[1]]))[0][0]
    
            sample_x_1 = X_pop[min_1]
            sample_x_2 = X_pop[min_2]
            obj_1_X.append(sample_x_1)
            obj_2_X.append(sample_x_2)
    
            sample_f_1 = F_pop[min_1]
            sample_f_2 = F_pop[min_2]
            obj_1_F.append(sample_f_1)
            obj_2_F.append(sample_f_2)
    
            #Need to find the samples that are the smallest away from the medians
            obj1_median = np.median(F_pop[:,[0]])
            obj2_median = np.median(F_pop[:,[1]])
    
            medians = np.hstack((obj1_median,obj2_median))
    
            mid_index1 = find_nearest(F_pop, medians)
            F_pop_new = np.delete(F_pop, mid_index1, axis = 0)
            mid_index2 = find_nearest(F_pop_new, medians)
    
            mid_X1.append(X_pop[mid_index1])
            mid_X2.append(X_pop[mid_index2])
    
        for i in range(len(obj_1_X)):
        
            fig, ax = plt.subplots()
    
            ax.contourf(XX, YY, ZZ, cmap = 'Blues', alpha = 0.2)
            colours = [[0.7,0.7,0.7]], [[0.5,0.5,0.5]]
            ax.set_facecolor((.95,.95,.95))
            for label in range(2):
                ax.scatter(x=X[Y==label, 0],
                           y=X[Y==label, 1],
                           c=colours[label],
                           s=40)
        
            prob_1 = model.predict_proba(obj_1_X[i].reshape(1,-1)).reshape(-1,1)
            prob_2 = model.predict_proba(obj_2_X[i].reshape(1,-1)).reshape(-1,1)
    
            prob_mid_1 = model.predict_proba(mid_X1[i].reshape(1,-1)).reshape(-1,1)
            prob_mid_2 = model.predict_proba(mid_X2[i].reshape(1,-1)).reshape(-1,1)
    
        
            #plt.scatter(x_orig[0], x_orig[1], c = 'green', marker = '*', s = 150, label = 'test_point') #original position
            '''Plotting the best objective 1 sample'''
            ax.scatter(obj_1_X[i][0], obj_1_X[i][1], c = [[0.,1,0.]], marker = '*', s = 200,
                       label = '{p1},{p2}'.format(p1 = prob_1[0], p2 = prob_1[1]))
    
            '''Plotting the best objective 2 sample'''
            ax.scatter(obj_2_X[i][0], obj_2_X[i][1], c = 'purple', marker = '*', s = 200,
                       label = '{p1},{p2}'.format(p1 = prob_2[0], p2 = prob_2[1]))
    
            '''Plotting a middle sample 1'''
            ax.scatter(mid_X1[i][0], mid_X1[i][1], c = [[.7,.4,.2]], marker = '*', s = 200,
                       label = '{p1},{p2}'.format(p1 = prob_mid_1[0], p2 = prob_mid_1[1]))
    
            '''Plotting a middle sample 2'''
            ax.scatter(mid_X2[i][0], mid_X2[i][1], c = 'orange', marker = '*', s = 200,
                       label = '{p1},{p2}'.format(p1 = prob_mid_2[0], p2 = prob_mid_2[1]))
    
    
            plt.legend(loc='upper right')
            plt.title('Generation: {g}'.format(g = i))
            plt.show()


#%%
    '''Plots pareto-front'''
    n_evals = []    # corresponding number of function evaluations\
    F_snapshots = []          # the objective space values in each generation
    cv = []         # constraint violation in each generation
    X_snapshots = []
    
    # iterate over the deepcopies of algorithms
    
    for algorithm in res2.history:
    
        # store the number of function evaluations
        n_evals.append(algorithm.evaluator.n_eval)
    
        # retrieve the optimum from the algorithm
        opt = algorithm.opt
    
        # store the least contraint violation in this generation
        cv.append(opt.get("CV").min())
    
        # filter out only the feasible and append
        _X = algorithm.pop.get("X")
        _F = algorithm.pop.get("F")
    
        X_snapshots.append(_X)
        F_snapshots.append(_F)
    
    for i in range(len(F_snapshots)):
    
        if i != len(F_snapshots)-1:
                pass
    
        else:
            current_snap = copy.deepcopy(F_snapshots[i])
        
            first_obj = (current_snap[:,[0]]).reshape(-1,1)
            second_obj = (current_snap[:,[1]]).reshape(-1,1)
            
            F_new = np.hstack((first_obj, second_obj))
        
            F_norm = current_snap
        
            R = np.random.uniform(low=0.0, high=1.0, size=None)
            G = np.random.uniform(low=0.0, high=1.0, size=None)
            B = np.random.uniform(low=0.0, high=1.0, size=None)
        
            colour = [[R, G, B]]
        
            if i == len(F_snapshots)-1:
                colour = [[0.,0.,0.]]
                plt.scatter(F_norm[:,[0]], F_norm[:,[1]], c = colour, s = 10)
        
            else:
                plt.scatter(F_norm[:,[0]], F_norm[:,[1]], c = colour, s = 10)
            #plt.title('Generation {g}'.format(g = i))
            plt.title('Pareto Front - Logistic - P')
            plt.xlabel('Objective 1')
            plt.ylabel('Objective 2')
            plt.show()
#%%
    #This section measures the optimization convergence and plots it

    obj1_cost = []
    obj2_cost = []
    for algorithm in res.history:
        _X = algorithm.pop.get("X")
        _F = algorithm.pop.get("F")
        
        obj1_cost.append(max(_F[:,[0]]))
        obj2_cost.append(max(_F[:,[1]]))
    
    gens = np.arange(0, len(obj1_cost))

    fig, (ax1, ax2) = plt.subplots(2,1)
    
    ax1.plot(gens, obj1_cost, c = 'red')
    ax1.set_xlim(0, 200)
    ax1.set_ylabel('Dist')
    ax1.set_title('Objective 1')
    
    ax2.plot(gens, obj2_cost, c = 'blue')
    ax2.set_xlim(0, 200)
    ax2.set_ylabel('Pred Diff')
    ax2.set_xlabel('Generations')
    ax2.set_title('Objective 2')
    
    fig.tight_layout()
    plt.show()
    
    obj1_cost = []
    obj2_cost = []
    for algorithm in res2.history:
        _X = algorithm.pop.get("X")
        _F = algorithm.pop.get("F")
        
        obj1_cost.append(max(_F[:,[0]]))
        obj2_cost.append(max(_F[:,[1]]))
    
    gens = np.arange(0, len(obj1_cost))

    fig, (ax1, ax2) = plt.subplots(2,1)
    
    ax1.plot(gens, obj1_cost, c = 'red')
    ax1.set_xlim(0, 200)
    ax1.set_ylabel('Dist')
    ax1.set_title('Objective 1')
    
    ax2.plot(gens, obj2_cost, c = 'blue')
    ax2.set_xlim(0, 200)
    ax2.set_ylabel('Pred Diff')
    ax2.set_xlabel('Generations')
    ax2.set_title('Objective 2')
    
    fig.tight_layout()
    plt.show()

#%%Need functions for
        
        
        