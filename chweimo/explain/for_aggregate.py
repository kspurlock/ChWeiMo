# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from chweimo.explain.weight import gini, kl_divergence


def aggregate_delta(
    section_X,
    section_F,
    section_X_orig,
    cm_section,
    col_name,
    feature_map,
    plausible,
    data_name,
):

    # Initialize empty arrays for changes and percent
    obj1_percent = np.empty((1, section_X_orig[0].shape[0]))
    obj2_percent = np.empty((1, section_X_orig[0].shape[0]))

    obj1_change = np.empty((1, section_X_orig[0].shape[0]))
    obj2_change = np.empty((1, section_X_orig[0].shape[0]))

    for i, orig_sample in enumerate(section_X_orig):

        x_subsection = section_X[i]  # Get final GA population
        f_subsection = section_F[i]  # Get final GA fitness

        obj_2 = f_subsection[:, [1]] * -1 * 100  # Convert fitness back to normal
        obj_1 = f_subsection[:, [0]] * -1

        # Find most optimal individuals
        obj1_best = x_subsection[np.where(obj_1 == max(obj_1))[0]]
        obj2_best = x_subsection[np.where(obj_2 == max(obj_2))[0]]

        # Calculate deltas and add to arrays
        obj1_deltas = obj1_best - orig_sample
        obj2_deltas = obj2_best - orig_sample

        obj1_change = np.vstack((obj1_change, obj1_deltas))
        obj2_change = np.vstack((obj2_change, obj2_deltas))

        # Find percentage change and add to arrays
        obj1_percent = abs(((obj1_deltas + 1) / (orig_sample + 1))) * 100
        obj2_percent = abs(((obj2_deltas + 1) / (orig_sample + 1))) * 100

        obj1_percent = np.vstack((obj1_percent, obj1_percent))
        obj2_percent = np.vstack((obj2_percent, obj2_percent))

    # Delete dummy entry from percents and find median
    obj1_percent = np.delete(obj1_percent, 0, axis=0)
    obj2_percent = np.delete(obj2_percent, 0, axis=0)

    obj1_percent = np.round(np.median(obj1_percent, axis=0), 2)
    obj2_percent = np.round(np.median(obj2_percent, axis=0), 2)

    # Delete dummy entry from changes and find median
    obj1_change = np.delete(obj1_change, 0, axis=0)
    obj2_change = np.delete(obj2_change, 0, axis=0)

    obj1_change = np.round(np.median(obj1_change, axis=0), 2)
    obj2_change = np.round(np.median(obj2_change, axis=0), 2)

    #######################################################################

    plot_delta(
        obj1_percent,
        obj2_percent,
        feature_map["continuous"],
        "Continuous",
        col_name,
        cm_section,
        plausible,
        data_name,
    )

    plot_delta(
        obj1_change,
        obj2_change,
        feature_map["discrete"],
        "Discrete",
        col_name,
        cm_section,
        plausible,
        data_name,
    )


def plot_delta(
    obj1, obj2, type_map, type_name, col_name, cm_section, plausible, data_name
):

    # Select features to display by type
    feature_names = copy.deepcopy(col_name[np.where(type_map == 1)])

    # Shorten column names
    for i, v in enumerate(feature_names):
        feature_names[i] = v[0:5]

    x_ticks = np.arange(0, len(feature_names))

    # Select individual features for specified map type
    obj1_features = obj1[np.where(type_map == 1)]
    obj2_features = obj2[np.where(type_map == 1)]

    # Construct bar plot
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x_ticks - width / 2, obj1_features, width, label="Obj1", color="tab:orange"
    )
    rects2 = ax.bar(
        x_ticks + width / 2, obj2_features, width, label="Obj2", color="tab:blue"
    )

    # Graph details
    plaus_title = "P" if plausible else "NP"
    ax.set_xlabel("Feature", fontdict={"fontsize": 12})
    ax.set_ylabel("Change", fontdict={"fontsize": 12})
    ax.set_title(
        "{d} - {t} Change Medians - {s} - {p}".format(
            d=data_name, t=type_name, s=cm_section, p=plaus_title
        ),
        fontdict={"fontsize": 12},
    )
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(feature_names, fontdict={"fontsize": 8})
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(linestyle="--", color="black")

    ax.bar_label(rects1, padding=3, fontsize=8)
    ax.bar_label(rects2, padding=3, fontsize=8)

    fig.tight_layout()
    plt.show()
    fig.savefig(
        "figures/{d}_{t}_change_{c}_{p}.png".format(
            d=data_name, t=type_name, c=cm_section, p=plaus_title
        )
    )


def aggregate_coeff(
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

        # Finding coefficients with GINI Grid Search
        clf = GridSearchCV(Lasso(), param_grid, scoring=gini)

        clf.fit(deltas, obj_2)

        gini_score = -1 * np.round(clf.best_score_, 2)

        coef = abs(clf.best_estimator_.coef_)
        coef_GINI = np.vstack((coef_GINI, coef))

        # Finding normal coefficients with default LASSO
        regressor = Lasso(alpha=0.01)  # Intialize
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

    plot_coeff(
        coef_KL, coef_GINI, coef_normal, col_name, cm_section, plausible, data_name
    )


def plot_coeff(
    coef_KL, coef_GINI, coef_normal, col_name, cm_section, plausible, data_name
):

    # Select features to display by type
    feature_names = copy.deepcopy(col_name)

    # Shorten column names
    for i, v in enumerate(feature_names):
        feature_names[i] = v[0:5]
        
    x_ticks = np.arange(0, len(coef_KL))

    # Construct bar plot
    width = 0.20
    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x_ticks - width/2,
        np.round(coef_normal, 2),
        width,
        label="No Grid Search",
        color="tab:purple",
    )
    rects2 = ax.bar(
        x_ticks + width/2,
        np.round(coef_KL, 2),
        width,
        label="With KL divergence",
        color="tab:blue",
    )
    
    """
    rects3 = ax.bar(
        x_ticks + (width/2)*2,
        np.round(coef_GINI, 2),
        width,
        label="With GINI",
        color="tab:orange",
    )
    """


    # Graph details
    plaus_title = "P" if plausible else "NP"
    ax.set_ylabel("Magnitude", fontdict={"fontsize": 12})
    ax.set_xlabel("Feature", fontdict={"fontsize": 12})

    ax.set_title(
        "{d} - Feature Weights - {s} - {p}".format(
            d=data_name, s=cm_section, p=plausible
        ),
        fontdict={"fontsize": 12},
    )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(feature_names, fontdict={"fontsize": 8})
    ax.legend(loc="upper left", fontsize=8)

    ax.bar_label(rects1, padding=3, fontsize=8)
    ax.bar_label(rects2, padding=3, fontsize=8)
    # ax.bar_label(rects3, padding=3, fontsize=8)

    fig.tight_layout()
    plt.show()
    fig.savefig("{d}_coef_{c}_{p}.png".format(d=data_name, c=cm_section, p=plaus_title))
