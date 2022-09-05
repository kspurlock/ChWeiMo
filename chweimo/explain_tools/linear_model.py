import itertools
import random
from math import log2

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class LassoGridSearch():
    def __init__(self, estimator, param_grid,
                 n_splits=5, n_combo=-1, refit=True):
        self.estimator_ = estimator
        self.param_grid_ = param_grid
        self.n_splits_ = n_splits
        self.n_combo_ = n_combo
        self.param_index_ = {k:i for i, k in enumerate(param_grid.keys())}
        self.results_ = {"param":[], "kl_divergence":[], "r2":[], "gini":[]}
        self.refit_ = refit
        
    def fit(self, X, y, lin_coef, verbose=False):
        random.seed(22222)
        combo = list(itertools.product(*self.param_grid_.values()))
        
        if self.n_combo_ > 0:
            combo = random.sample(combo, self.n_combo_)
        else:
            combo = random.sample(combo, len(combo))
        
        for c in combo:
            self.results_["param"].append(c)
            
            kl_scores, r2_scores, gini_scores = [], [], []
            
            # Setting up current coefficients
            for k, i in self.param_index_.items():
                vars(self.estimator_)[k] = c[self.param_index_[k]]
                
        
            for train_ind, test_ind in KFold(self.n_splits_).split(X, y):
                X_train, X_test = X[train_ind], X[test_ind]
                y_train, y_test = y[train_ind], y[test_ind]
                
                self.estimator_.fit(X_train, y_train)
                
                lasso_coef = self.estimator_.coef_
                yh = self.estimator_.predict(X_test)
                
                kl_d = kl_divergence(lin_coef, lasso_coef)
                r2 = r2_score(y_test, yh)
                g = gini(lasso_coef)
                
                kl_scores.append(kl_d)
                r2_scores.append(r2)
                gini_scores.append(g)
            
            self.results_["kl_divergence"].append(np.mean(kl_scores))
            self.results_["r2"].append(np.mean(r2_scores))
            self.results_["gini"].append(np.mean(gini_scores))
            
        if self.refit_:
            #combined_obj = np.multiply(self.results_["kl_divergence"], self.results_["gini"])
            #combined_obj = self.results_["kl_divergence"] + self.results_["gini"] + self.results_["r2"]
            r2 = np.array(self.results_["r2"], dtype="float64")
            combined_obj = np.add(r2, self.results_["gini"])
            combined_obj = np.add(combined_obj, self.results_["kl_divergence"])
            
            best = np.argmax(combined_obj)
            
            if verbose:
                print("R2: {}".format(self.results_["r2"][best]))
                print("KL Divergence: {}".format(self.results_["kl_divergence"][best]))
                print("gini: {}".format(self.results_["gini"][best]))
            
            for k, i in self.param_index_.items():
                vars(self.estimator_)[k] = self.results_["param"][i][self.param_index_[k]]
                
            self.estimator_.fit(X, y)
            
        return

def kl_divergence(coef_P, coef_Q):
    P = abs(coef_P)+1
    Q = abs(coef_Q)+1
    
    prob_P = P/np.sum(P)
    prob_Q = Q/np.sum(P)
    
    d = 0
    for i in range(len(prob_P)):
        # Reasoning for q/p formulation is that it is more likely
        # for Q(i) to be ~0 than for P(i) to be (due to sparsity)
        d += prob_P[i]*log2(prob_Q[i]/prob_P[i])
    
    return (1/(1+np.exp(d)))

def gini(coef):
    C = np.sort(abs(coef)+1)
    N = len(C)
    l1 = np.sum(C)
    
    summation = 0
    for i in range(N):
        a = C[i]/l1
        b = (N - (i+1) + 0.5)/N
        
        summation += np.sum(a*b)
        
    total = 1 - 2*summation
    
    return total

def find_weight(
    explainer, **kwargs
):
    res = explainer.get_results()
    X_pop = res.history[-1].pop.get("X")
    F_pop = res.history[-1].pop.get("F")
    
    obj_2 = F_pop[:, [1]] * -1 * 100
    
    #coef_KL = np.full((X_pop.shape[0], X_pop.shape[1]), 0)
    #coef_normal = np.full((X_pop.shape[0], X_pop.shape[1]), 0)

    deltas = X_pop - explainer.sample_
    deltas = StandardScaler().fit_transform(deltas)
    
    ###############################################################
    #      Finding coefficients by normal Linear Model            #
    ###############################################################
    
    linear_coef = LinearRegression().fit(deltas, obj_2).coef_
    linear_coef = abs(linear_coef.reshape(-1,))
    
    ###############################################################
    #              Finding coefficients with LASSO                #
    ###############################################################
    
    param_grid = {"alpha": np.linspace(.1, 5, 15),
                  "C": np.linspace(0, 10, 15)}

    clf = LassoGridSearch(Lasso(random_state=22222), param_grid=param_grid, n_combo=-1)
    
    if "verbose" in kwargs.keys():
        clf.fit(deltas, obj_2, linear_coef, verbose=kwargs["verbose"])
    else:
        clf.fit(deltas, obj_2, linear_coef)
        
    lasso_coef = abs(clf.estimator_.coef_)
    
    return {"normal": linear_coef, "sparse": lasso_coef}

def consolidate_agg_weight(agg):
    coef_dict = {"normal":[], "sparse":[]}
    
    for i in agg:
        for k in coef_dict:
            coef_dict[k].append(i[k])
    
    for k in coef_dict:
        coef_dict[k] = np.median(coef_dict[k], axis=0)
    
    return coef_dict