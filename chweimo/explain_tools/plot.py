import copy
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
from chweimo.explain_tools.change import find_changes
from chweimo.explain_tools.linear_model import find_weight


def plot_weight(
    coef_dict, col_names, **kwargs
):
    normal_coef = coef_dict["normal"]
    sparse_coef = coef_dict["sparse"]
    
    # Select features to display by type
    short_col = copy.deepcopy(col_names)

    # Shorten column names
    for i, v in enumerate(short_col):
        short_col[i] = v[0:5]

    ticks = np.arange(0, len(sparse_coef))

    # Construct bar plot
    width = 0.5
    
    fig, ax = plt.subplots()
    if "barh" in kwargs.keys() and kwargs["barh"]==True:
        
        rects1 = ax.barh(
            ticks,
            np.round(normal_coef, 2),
            width,
            label="Normal Linear",
            color="darkgray",
        )
        rects2 = ax.barh(
            ticks,
            np.round(sparse_coef, 2),
            width,
            label="Sparsified (Grid Search)",
            color="tab:cyan",
        )
        
        ax.set_xlabel("Magnitude", fontdict={"fontsize": 12})
        ax.set_ylabel("Feature", fontdict={"fontsize": 12})

        ax.set_yticks(ticks)
        ax.set_yticklabels(short_col, fontdict={"fontsize": 8})
        ax.legend(loc="best", fontsize=8)
    
    else:
        rects1 = ax.bar(
            ticks,
            np.round(normal_coef, 2),
            width,
            label="Normal Linear",
            color="darkgray",
        )
        rects2 = ax.bar(
            ticks,
            np.round(sparse_coef, 2),
            width,
            label="Sparsified (Grid Search)",
            color="tab:cyan",
        )
        
        ax.set_ylabel("Magnitude", fontdict={"fontsize": 12})
        ax.set_xlabel("Feature", fontdict={"fontsize": 12})

        ax.set_xticks(ticks)
        ax.set_xticklabels(short_col, fontdict={"fontsize": 8})
        ax.legend(loc="upper left", fontsize=8)

    if "bar_label" in kwargs.keys() and kwargs["bar_label"]==True:
        ax.bar_label(rects1, fontsize=8, label_type="edge")
        ax.bar_label(rects2, fontsize=8, label_type="center")

    fig.tight_layout()
    
    return (fig, ax)

def plot_change(change_dict, type_dict, type_name, col_names, **kwargs):
    short_names = copy.deepcopy(col_names[np.where(type_dict[type_name]==1)])
    
    if len(short_names) == 0:
        print("Could not plot changes for {}. type_dict indicates no features of this type.".format(type_name))
        return (0, 0)
    
    else:
        for i, v in enumerate(short_names):
            short_names[i] = v[0:5]
            
        ticks = np.arange(0, len(short_names))
        
        if type_name == "continuous":
            obj1_features = change_dict["change"]["obj1"]
            obj2_features = change_dict["change"]["obj2"]
        elif type_name == "discrete":
            obj1_features = change_dict["percent"]["obj1"]
            obj2_features = change_dict["percent"]["obj2"]
        else:
            raise ValueError("Invalid type_name")
        
        print(obj1_features, np.where(type_dict[type_name]==1))
        
        obj1_features = obj1_features[np.where(type_dict[type_name]==1)]
        
        obj2_features = obj2_features[np.where(type_dict[type_name]==1)]
        
        width = 0.35
        
        fig, ax = plt.subplots()
        if "barh" in kwargs.keys() and kwargs["barh"]==True:
            
            rects1 = ax.barh(
                ticks - width / 2, obj1_features, width, label="Obj1", color="tab:orange"
            )
            rects2 = ax.barh(
                ticks + width / 2, obj2_features, width, label="Obj2", color="tab:blue"
            )
            
            ax.set_ylabel("Feature", fontdict={"fontsize": 12})
            ax.set_xlabel("{} Change".format("%" if type_name=="continuous" else ""), fontdict={"fontsize": 12})
            ax.set_yticks(ticks)
            ax.set_yticklabels(short_names, fontdict={"fontsize": 8})
            ax.legend(loc="best", fontsize=8)
            ax.axvline(linestyle="--", color="black")
            
        else:
            fig, ax = plt.subplots()
            rects1 = ax.bar(
                ticks - width / 2, obj1_features, width, label="Obj1", color="tab:orange"
            )
            rects2 = ax.bar(
                ticks + width / 2, obj2_features, width, label="Obj2", color="tab:blue"
            )
            
            ax.set_xlabel("Feature", fontdict={"fontsize": 12})
            ax.set_ylabel("{} Change".format("%" if type_name=="continuous" else ""), fontdict={"fontsize": 12})
            ax.set_xticks(ticks)
            ax.set_xticklabels(short_names, fontdict={"fontsize": 8})
            ax.legend(loc="best", fontsize=8)
            ax.axhline(linestyle="--", color="black")
            
        # Graph details
        if "bar_label" in kwargs.keys() and kwargs["bar_label"]==True:
            ax.bar_label(rects1, padding=3, fontsize=8)
            ax.bar_label(rects2, padding=3, fontsize=8)

        fig.tight_layout()
    
    return (fig, ax)

def show_change_weights(explainer, **kwargs):
    verbose = kwargs["verbose"] if "verbose" in kwargs else False
    coef_dict = find_weight(explainer, verbose=verbose)
    fig, ax = plot_weight(coef_dict, explainer.col_names_, **kwargs)
    
    return (fig, ax)

def show_change(explainer, type_dict, **kwargs):
    change_dict = find_changes(explainer)
    fig1, ax1 = plot_change(change_dict, type_dict, "continuous", explainer.col_names_, **kwargs)
    fig2, ax2 = plot_change(change_dict, type_dict, "discrete", explainer.col_names_, **kwargs)
    
    return fig1, ax1, fig2, ax2

def design_title(type_name, fig_context, data_name, section):
    return "{d}-{t}{f}{s}".format(d=data_name,
                                           t=type_name,
                                           s=section,
                                           f=fig_context)
    
def figure_designer(fig, ax, type_name, fig_context, plausible, data_name, section):
    pad = 25 if plausible else 0
    title = design_title(type_name, fig_context, data_name, section)
    ax.set_title(title, pad=pad)
    
    if plausible:
        fig.text(.5, 0.98, "Plausible Results", ha="center")
        
    fig.savefig("{d}{t}.png".format(d=date.today(), t=title), bbox_inches="tight")
    
    return fig
