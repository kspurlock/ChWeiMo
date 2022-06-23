import numpy as np
import copy

def find_samples(x_test, model, borderline):
    """Borderline is a bool var that determines whether to find confident or 
        borderline samples based on predictions"""

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