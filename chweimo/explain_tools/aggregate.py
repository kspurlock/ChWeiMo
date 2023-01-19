import numpy as np
from chweimo.explain_tools.change import consolidate_agg_change, find_changes
from chweimo.explain_tools.linear_model import (consolidate_agg_weight,
                                                find_weight)
from chweimo.explain_tools.plot import figure_designer
from chweimo.explain_tools.plot import plot_change, plot_weight
from tqdm import tqdm

def perform_aggregation(explainer, cm_splits, type_dict,
                        use_MAD=True, plausible=True, data_name="", **kwargs):
    # -explainer is reused for its settings
    for section in cm_splits["samples"]: #true_neg, false_neg, etc.
        change_agg = []
        weight_agg = []
        
        for i, sample in enumerate(cm_splits["samples"][section]):
            print(section)
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
            
        if len(change_agg) != 0:
            merged_change = consolidate_agg_change(change_agg)
            merged_weight = consolidate_agg_weight(weight_agg)

            fig, ax = plot_change(merged_change, type_dict, "continuous", explainer.col_names_, **kwargs)
            if fig != 0:
                figure_designer(fig, ax, " Continuous ", "Change Medians", plausible, data_name, " - " + section)
                
            fig, ax = plot_change(merged_change, type_dict, "discrete", explainer.col_names_, **kwargs)
            if fig != 0:
                figure_designer(fig, ax, " Discrete ", "Change Medians", plausible, data_name, " - " + section)
                
            fig, ax = plot_weight(merged_weight, explainer.col_names_, **kwargs)
            figure_designer(fig, ax, "", " Feature Weights", plausible, data_name, " - " + section)
        
        else:
            print("Section '{}' did not have any samples, skipping...".format(section))
            pass
        
        if "debug" in kwargs and kwargs["debug"] == True:
            break
    
        # if delta_heatmap:
            # do delta_heatmap(change_agg)
            # although I will have to go back and get a delta agg
    
    return
