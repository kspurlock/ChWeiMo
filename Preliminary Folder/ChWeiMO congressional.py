# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 18:00:11 2021

@author: kylei
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:03:21 2021

@author: kylei
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from Explainer import Explainer

def plot_sample(sample, title, prediction, c):
    fig, ax = plt.subplots()

    ax.contourf(XX, YY, ZZ, cmap = 'Blues', alpha = 0.2)

    ax.set_facecolor((.95,.95,.95))
    colours = [[0.7,0.7,0.7]], [[0.5,0.5,0.5]]

    for label in range(2): #Range 2 here is the number of classes
        ax.scatter(x=X[Y==label, 0],
                   y=X[Y==label, 1],
                   c=colours[label],
                   s=40,
                   label='Class {c}'.format(c = label))

    plt.scatter(sample[0], sample[1], c = c, marker = '*', s = 150,
                label = '{p1}, {p2}'.format(p1 = prediction.reshape(-1)[0],
                                            p2 = prediction.reshape(-1)[1]))

    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


def find_contour(X, y, model):
    #Define the bounds of features
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    #Create lines and rows of grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    #Transform grid into a vector
    r1, r2 = xx.flatten(), yy.flatten() #Flatten grids
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    #Stack the vectors to create inputs to model
    grid = np.hstack((r1,r2))

    #Fit and get predictions for model
    model = model
    model.fit(X, y)
    yhat = model.predict(grid)

    #Reshape predictions into a grid
    zz = yhat.reshape(xx.shape)

    #Return the all grid rows to plot contour
    return xx, yy, zz

def preprocessing():
    dataset = pd.read_csv('house-votes-84.csv').replace('?', np.nan)
    
    dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    lb = LabelEncoder()
    
    for column in dataset.columns:
        if dataset[column].dtype != np.dtype('int64'):
            dataset[column] = lb.fit_transform(dataset[column])
            
    return dataset

def preprocessing_j():
    '''Jaylen's preprocessing mode'''
    dataset = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data")
    dataset.columns = ['Party', 'Handicapped Infants', 'Water Project Cost Sharing', 'Budget Resolution',
                       'Physician Fee Freeze', 'El Salvador Aid', 'Religious Groups in School',
                       'Anti-Satellite Test Ban', 'Aid to Nicaraguan Contras', 'Mx Missile', 'Immigration',
                       'Synfuels Corporation Cutback', 'Education Spending', 'Superfund Right to Sue', 'Crime',
                       'Duty-Free Exports', 'Export Administration Act']
    feature_names = ['Handicapped Infants', 'Water Project Cost Sharing', 'Budget Resolution',
                     'Physician Fee Freeze', 'El Salvador Aid', 'Religious Groups in School',
                     'Anti-Satellite Test Ban', 'Aid to Nicaraguan Contras', 'Mx Missile', 'Immigration',
                     'Synfuels Corporation Cutback', 'Education Spending', 'Superfund Right to Sue', 'Crime',
                     'Duty-Free Exports', 'Export Administration Act']
    class_name = ['Party']

    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')  # imputer to fill missing values with most
    # common outcome
    le = LabelEncoder()
    dataset.replace('?', np.nan, inplace=True)  # replaces ? in dataset with np.nan for imputer
    for col in dataset.columns:
        dataset[col] = si.fit_transform(dataset[col].values.reshape(-1, 1))  # imputer filling missing values in each
        # column
    for col in dataset.columns:
        dataset[col] = le.fit_transform(dataset[col])

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

            metrics.append(confusion_matrix(y_test, y_pred))

            metric_dict[it] = metrics
            split_dict[it] = (train_ind, test_ind)
            it += 1
            
        return total_cm, metric_dict, split_dict

def split_by_cm(x, y, model):
    true_neg = [] #C 0,0
    false_neg = [] #C 1, 0
    true_pos = [] #C 1, 1
    false_pos = [] #C 0, 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify = y)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.rcParams['font.size'] = '20'
    print(confusion_matrix(y_test, y_pred))
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

'''Begin main execution'''
if __name__ == '__main__':
    global input_shape

    dataset = preprocessing_j()

    #model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    model = DecisionTreeClassifier()

    cols = dataset.iloc[:,1:].columns
    X = dataset.drop(dataset.columns[0], axis = 1).values
    Y = dataset.iloc[:, [0]].values.reshape(-1,)

    input_shape = X.shape[1]

    cm_splits, x_train, x_test, y_train, y_test = split_by_cm(X, Y, model)

#%%
    '''Begin optimization section'''
    full_results = []
    full_pred = []
    full_x_orig = []

    start = time.process_time() #Check how long non-plausible takes
    for lst in cm_splits: #lst is each of the cm sections
        section_F = []
        section_X = []
        section_X_orig = []
        section_pred = []

        for i in lst: #sample in each of the confusion matrix
            x_orig = i
            x_orig_y = model.predict_proba(x_orig.reshape(1,-1)).reshape(-1)
            print(i)

            change_class = np.argmin(x_orig_y) #what class do we need to change to
            section_pred = np.append(section_pred, change_class)

            explainer = Explainer(X, Y, model)
            res = explainer.explain_instance(sample = x_orig,
                                             change_class = change_class,
                                             plausible = True,
                                             method = 'NSGA2',
                                             method_params = None)

            section_X = np.append(section_X, res.history[-1].pop.get("X"))
            section_F = np.append(section_F, res.history[-1].pop.get("F"))
            section_X_orig = np.append(section_X_orig, i)
            section_pred = np.append(section_pred, x_orig_y)
            
        section_F_X = np.array([section_X, section_F])
        full_results = np.append(full_results, section_F_X)
        full_pred = np.append(full_pred, section_pred)
        full_x_orig = np.append(full_x_orig, section_X_orig)
    stop = time.process_time() - start

#%%
    '''Note to self, all coefficients are zero because all changes are the same, and all y_pred are the same as well'''
    '''Maybe need to combine false pos and false negative to truly see interclass feature influence'''
    from sklearn.linear_model import LinearRegression, Lasso

    def find_change_weight(opt_result):
        for algorithm in opt_result:
            F_pop = algorithm.pop.get("F") #Get objective values
            X_pop = algorithm.pop.get('X') #Get pop samples
            obj_2 = F_pop[:,[1]]*-1*100
            obj_1 = F_pop[:,[0]]*-1
    
            deltas = []
            for i in range(X_pop.shape[0]):
                deltas.append(X_pop[i] - x_orig)
    
            deltas = np.array(deltas)

            '''Fit the regressor to delta and obj2 and get coefficients'''
            regressor = Lasso(alpha=0.2) #Intialize
            regressor.fit(deltas, obj_2) #Fit
    
            coef = regressor.coef_.reshape(-1,1) #Coefficients
            '''
            for i, j in zip(cols, coef): #Output coefficient per feature
                print('{c}: {co}'.format(c = i, co = np.round(j[0],4)))
            '''
            coef = abs(coef.reshape(-1))
            dct = {}
            for i in range(coef.shape[0]):
                dct[cols[i]] = coef[i]

            return dct

    full_coef = []
    #Now need to find individual coefficient weights per algorithm object
    for lst in full_results: #for result in split
        section = []
        for i in lst: #for algorithm instance in lst:
            coef = find_change_weight(i)
            section.append(coef)
        full_coef.append(section)

    final_coef = []
    it = 0
    for lst in full_coef: #These are the 4 sections of confusion matrix
        '''Takes me into an array of dictionaries'''
        section = []
        feature_vals = {key: [] for key in cols}
        for dct in lst: #These are the dcts of solutions
            for key in cols:
                feature_vals[key].append(dct[key])
        for key in cols:
            feature_vals[key] = np.median(feature_vals[key])

        final_coef.append(feature_vals)

#%%
    import lime

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(x_train,
                                                            feature_names=cols,
                                                            discretize_continuous=True,
                                                            mode = 'classification')

    split_index = 1

    it = 0
    lime_TP_scores = []

    for i in cm_splits[split_index]:
        exp = lime_explainer.explain_instance(i, model.predict_proba)
        fig = exp.as_list()

        print(it/len(cm_splits[split_index]))
        it+=1

        new_lime_scores = {key: [] for key in cols}
        seen_names = [] #everytime a name is changed, put it in this array
        for name1, val in fig: #name1 is the weird name LIME comes up with
            for name2 in cols: #name2 is the name we want
                if not name1.find(name2) == -1:
                    new_lime_scores[name2] = abs(val)
                    seen_names.append(name2)
                else:
                    pass

        for name2 in cols:
            if name2 not in seen_names:
                new_lime_scores[name2] = 0

        lime_TP_scores.append(new_lime_scores)

    lime_feature_vals = {key: [] for key in cols}
    for dct in lime_TP_scores:
        for key in cols:
            lime_feature_vals[key].append(dct[key])
    for key in cols:
        lime_feature_vals[key] = np.median(lime_feature_vals[key])

#%%
    import math

    def convert(lst):
        dct = {lst[i][0]: lst[i][1] for i in range(0, len(lst))}
        return dct

    def kendall_tau(list1, list2):
        keys = [i for i in list1.keys()]
        concordant_pairs = 0
        discordant_pairs = 0
        group1_ties = 0
        group2_ties = 0
    
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if i == j:
                    pass
                else:
                    feature_1 = keys[i]
                    feature_2 = keys[j]
    
                    list_1_v1 = list1[feature_1]
                    list_1_v2 = list1[feature_2]
    
                    list_2_v1 = list2[feature_1]
                    list_2_v2 = list2[feature_2]
    
                    if list_1_v1 > list_1_v2 and list_2_v1 > list_2_v2:
                        concordant_pairs += 1 
    
                    elif list_1_v1 < list_1_v2 and list_2_v1 < list_2_v2:
                        concordant_pairs += 1
                       
                    elif list_1_v1 < list_1_v2 and list_2_v1 > list_2_v2:
                        discordant_pairs += 1
                        
                    elif list_1_v1 > list_1_v2 and list_2_v1 < list_2_v2:
                        discordant_pairs += 1
                       
                    if list_1_v1 == list_1_v2:
                        group1_ties += 1
                     
                    if list_2_v1 == list_2_v2:
                        group2_ties += 1

        print("Number of concordant pairs: ", concordant_pairs)
        print("Number of discordant pairs: ", discordant_pairs)
        print("Number of ties in group 1: ", group1_ties)
        print("Number of ties in group 2: ", group2_ties)
        n0 = (len(list1) * (len(list1) - 1)) / 2
        taua_coefficient = (concordant_pairs - discordant_pairs) / n0
        taub_coefficient = (concordant_pairs - discordant_pairs) / math.sqrt((n0 - group1_ties) * (n0 - group2_ties))
        print("Kendall tau-a coefficient: ", taua_coefficient)
        print("Kendall tau-b coefficient: ", taub_coefficient)
        return taua_coefficient, taub_coefficient



    counterfactual_TP = final_coef[split_index]
    counterfactual_TP = sorted(counterfactual_TP.items(), key=lambda x:x[1], reverse=True)
    counterfactual_TP = convert(counterfactual_TP)

    lime_TP = sorted(lime_feature_vals.items(), key=lambda x:x[1], reverse=True)
    lime_TP = convert(lime_TP)

    kendall_tau(lime_TP, counterfactual_TP)


#%%
    '''Plotting'''

    ticks = list(counterfactual_TP.keys())

    for i in range(len(ticks)):
        ticks[i] = ticks[i][0:3]


    plt.bar(np.arange(0,len(counterfactual_TP.keys())), counterfactual_TP.values(), width = 0.5, color = 'tab:blue')
    plt.xticks(np.arange(0,len(counterfactual_TP.keys())), ticks, fontsize=9)
    plt.title('Counterfactual Feature Weights - FN')
    plt.xlabel('Coefficient/Feature')
    plt.ylabel('Feature Weight')
    plt.show()

    ticks_LIME = list(lime_TP.keys())

    for i in range(len(ticks_LIME)):
        ticks_LIME[i] = ticks_LIME[i][0:3]

    plt.bar(np.arange(0,len(lime_TP.keys())), lime_TP.values(), width = 0.5, color = 'tab:blue')
    plt.xticks(np.arange(0,len(lime_TP.keys())), ticks_LIME, fontsize=9)
    plt.title('LIME Feature Weights - FN')
    plt.xlabel('Coefficient/Feature')
    plt.ylabel('Feature Weight')
    plt.show()

    '''Maybe combine these into a four section plot'''