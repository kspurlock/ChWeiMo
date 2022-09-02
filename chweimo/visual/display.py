# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import copy
import numpy as np


def plot_cf(
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


def plot_weight(
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
        x_ticks - width / 2,
        np.round(coef_normal, 2),
        width,
        label="No Grid Search",
        color="tab:purple",
    )
    rects2 = ax.bar(
        x_ticks + width / 2,
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
    fig.savefig(
        "figures/{d}_coef_{c}_{p}.png".format(d=data_name, c=cm_section, p=plaus_title))
