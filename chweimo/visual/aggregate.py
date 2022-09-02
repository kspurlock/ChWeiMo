# -*- coding: utf-8 -*-
from importlib.metadata import requires
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from chweimo.explain.linear_sparsity import kl_divergence
from chweimo.explain.display import plot_cf, plot_weight


def explain_single(optimizer, feature_map, col_name, **kwargs):
    res_ = optimizer.res_
    
    section_X = [res_.history[-1].pop.get("X")]
    section_F = [res_.history[-1].pop.get("F")]
    section_X_orig = [optimizer.sample_]
    section_pred = [optimizer.change_class_]
        
    plausible = optimizer.plausible_
    data_name = kwargs["data_name"] if "data_name" in kwargs.keys() else ""
    
    aggregate_cf(section_X, section_F, section_X_orig, "", col_name, feature_map, plausible, data_name)
    aggregate_weight(section_X, section_F, section_X_orig, cm_section, col_name, plausible, data_name)
    
    return


def aggregate_cf(
    section_X,
    section_F,
    section_X_orig,
    cm_section,
    col_name,
    feature_map,
    plausible,
    data_name,
):

    # Initialize empty dicts of arrays for changes, percents
    obj_percent = {}
    obj_change = {}
    
    for key in ["obj1", "obj2"]:
        obj_percent[key] = np.empty((1, section_X_orig[0].shape[0]))
        obj_change[key] = np.empty((1, section_X_orig[0].shape[0]))

    
    for i, orig_sample in enumerate(section_X_orig):

        x_subsection = section_X[i]  # Get final GA population
        f_subsection = section_F[i]  # Get final GA fitness
        
        obj_1 = f_subsection[:, [0]] * -1 # Convert fitness back to normal
        obj_2 = f_subsection[:, [1]] * -1 * 100  
        
        # Find most optimal individuals
        obj1_best = x_subsection[np.where(obj_1 == max(obj_1))[0][0]]
        obj2_best = x_subsection[np.where(obj_2 == max(obj_2))[0][0]]
        print(f"Length: {len(obj2_best)})")

        # Calculate deltas and add to arrays
        obj1_cf = obj1_best - orig_sample
        obj2_cf = obj2_best - orig_sample

        obj_change["obj1"] = np.vstack((obj_change["obj1"], obj1_cf))
        obj_change["obj2"] = np.vstack((obj_change["obj2"], obj2_cf))

        # Find percentage change and add to arrays
        obj1_percent = abs(((obj1_cf + 1) / (orig_sample + 1))) * 100
        obj2_percent = abs(((obj2_cf + 1) / (orig_sample + 1))) * 100

        obj_percent["obj1"] = np.vstack((obj_percent["obj1"], obj1_percent))
        obj_percent["obj2"] = np.vstack((obj_percent["obj2"], obj2_percent))

    # Delete dummy entry from percents and find median
    obj_percent["obj1"] = np.delete(obj_percent["obj1"], 0, axis=0)
    obj_percent["obj2"] = np.delete(obj_percent["obj2"], 0, axis=0)

    obj1_percent_med = np.round(np.median(obj_percent["obj1"], axis=0), 2)
    obj2_percent_med = np.round(np.median(obj_percent["obj2"], axis=0), 2)

    # Delete dummy entry from changes and find median
    obj_change["obj1"] = np.delete(obj_change["obj1"], 0, axis=0)
    obj_change["obj2"] = np.delete(obj_change["obj2"], 0, axis=0)

    obj1_change_med = np.round(np.median(obj_change["obj1"], axis=0), 2)
    obj2_change_med = np.round(np.median(obj_change["obj2"], axis=0), 2)

    # Plot median changes and percentages based on feature type
    
    plot_cf(
        obj1_percent_med,
        obj2_percent_med,
        feature_map["continuous"],
        "Continuous",
        col_name,
        cm_section,
        plausible,
        data_name,
    )

    plot_cf(
        obj1_change_med,
        obj2_change_med,
        feature_map["discrete"],
        "Discrete",
        col_name,
        cm_section,
        plausible,
        data_name,
    )
    
    return obj_change # Return counterfactuals

def aggregate_weight(
    section_X, section_F, section_X_orig, cm_section, col_name, plausible, data_name
):

    coef_KL = np.empty((1, section_X_orig[0].shape[0]))
    coef_normal = np.empty((1, section_X_orig[0].shape[0]))
    coef_GINI = np.empty((1, section_X_orig[0].shape[0]))

    for i, orig_sample in enumerate(section_X_orig):
        x_subsection = section_X[i]
        f_subsection = section_F[i]

        obj_2 = f_subsection[:, [1]] * -1 * 100  # Converting obj2 back to normal

        deltas = []

        for x_new in x_subsection:
            deltas.append(x_new - orig_sample)

        deltas = StandardScaler().fit_transform(np.array(deltas))

        # Finding coefficients with KL Grid Search
        param_grid = {"alpha": np.linspace(0.1, 2, 20)}
        clf = GridSearchCV(Lasso(), param_grid, scoring=kl_divergence)

        clf.fit(deltas, obj_2)

        kl_divergence_score = -1 * np.round(clf.best_score_, 2)

        coef = abs(clf.best_estimator_.coef_)
        coef_KL = np.vstack((coef_KL, coef))
        
        """
        # Finding coefficients with GINI Grid Search
        clf = GridSearchCV(Lasso(), param_grid, scoring=gini)

        clf.fit(deltas, obj_2)

        gini_score = -1 * np.round(clf.best_score_, 2)

        coef = abs(clf.best_estimator_.coef_)
        coef_GINI = np.vstack((coef_GINI, coef))
        """
        
        # Finding normal coefficients with default LASSO
        regressor = LinearRegression()  # Intialize
        regressor.fit(deltas, obj_2)  # Fit
        coef = abs(regressor.coef_.reshape(1, -1))  # Coefficients
        coef_normal = np.vstack((coef_normal, coef))

    # Finding median of coefficients for KL grid search
    coef_KL = np.delete(coef_KL, 0, axis=0)
    coef_KL = np.median(coef_KL, axis=0)

    # Finding median of coefficients for GINI grid search
    coef_GINI = np.delete(coef_GINI, 0, axis=0)
    coef_GINI = np.median(coef_GINI, axis=0)

    # Finding median of coefficients WITHOUT grid search
    coef_normal = np.delete(coef_normal, 0, axis=0)
    coef_normal = np.median(coef_normal, axis=0)

    """
    # GINI for KL grid search coefficients and Normal coefficients
    sparsity_measure_KL = np.round(gini(coef_median_KL.reshape(-1,1)),2)
    sparsity_measure_normal = np.round(gini(coef_median_normal.reshape(-1,1)),2)
    """

    plot_weight(
        coef_KL, coef_GINI, coef_normal, col_name, cm_section, plausible, data_name
    )