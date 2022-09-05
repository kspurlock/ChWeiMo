from datetime import date

import numpy as np
from chweimo.explain_tools.change import consolidate_agg_change, find_changes
from chweimo.explain_tools.linear_model import (consolidate_agg_weight,
                                                find_weight)
from chweimo.explain_tools.plot import plot_change, plot_weight
from tqdm import tqdm


def design_title(type_name, fig_context, data_name, section):
    return "{d} - {t}{f} - {s}".format(d=data_name,
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

def perform_aggregation(explainer, cm_splits, type_dict,
                        use_MAD=True, plausible=True, data_name="", **kwargs):
    # -explainer is reused for its settings
    for section in cm_splits["samples"]: #true_neg, false_neg, etc.
        change_agg = []
        weight_agg = []
        
        for i, sample in tqdm(enumerate(cm_splits["samples"][section])):
            sample_y = cm_splits["prob"][section][i]
            
            change_class = np.argmin(sample_y)
            
            explainer.generate_cf(
                sample=sample,
                change_class=change_class,
                plausible=plausible,
                use_MAD=use_MAD,
                **kwargs
            )
            
            change = find_changes(explainer)
            weight = find_weight(explainer)
            change_agg.append(change)
            weight_agg.append(weight)
            
        
        merged_change = consolidate_agg_change(change_agg)
        merged_weight = consolidate_agg_weight(weight_agg)

        fig, ax = plot_change(merged_change, type_dict, "continuous", explainer.col_names_)
        figure_designer(fig, ax, "Continuous ", "Change Medians", plausible, data_name, section)
            
        fig, ax = plot_change(merged_change, type_dict, "discrete", explainer.col_names_)
        figure_designer(fig, ax, "Discrete ", "Change Medians", plausible, data_name, section)
            
        fig, ax = plot_weight(merged_weight, explainer.col_names_)
        figure_designer(fig, ax, "", "Feature Weights", plausible, data_name, section)
        break

        
    return merged_change
