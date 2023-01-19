import numpy as np

def find_changes(
    explainer
):
    obj_change = {"obj1":None, "obj2":None}
    obj_percent = {"obj1":None, "obj2":None}
    
    res = explainer.get_results()
    orig_sample = explainer.sample_
    
    X_pop = res.pop.get("X")
    F_pop = res.pop.get("F")
    
    obj_1 = F_pop[:, [0]] * -1
    obj_2 = F_pop[:, [1]] * -1 * 100
    
    obj1_best = X_pop[np.where(obj_1 == max(obj_1))[0][0]]
    obj2_best = X_pop[np.where(obj_2 == max(obj_2))[0][0]]
    
    # Find changes and percentages (discrete and continuous)
    obj1_change = obj1_best - orig_sample
    obj2_change = obj2_best - orig_sample
    
    obj1_percent = abs(((obj1_change + 1) / (orig_sample + 1))) * 100
    obj2_percent = abs(((obj2_change + 1) / (orig_sample + 1))) * 100
    
    obj_change["obj1"] = obj1_change
    obj_change["obj2"] = obj2_change
    obj_percent["obj1"] = obj1_percent
    obj_percent["obj2"] = obj2_percent
                
    return {"change":obj_change, "percent": obj_percent}

def consolidate_agg_change(agg):
    
    obj_change = {"obj1":[], "obj2":[]}
    obj_percent = {"obj1":[], "obj2":[]}
    
    for i in agg:
        for k in obj_change:
            obj_change[k].append(i["change"][k])
            obj_percent[k].append(i["percent"][k])
            
    for k in obj_change:
        obj_change[k] = np.round(np.median(obj_change[k], axis=0), 2)
        obj_percent[k] = np.round(np.median(obj_percent[k], axis=0), 2)
        
    return {"change":obj_change, "percent":obj_percent}