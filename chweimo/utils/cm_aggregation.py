# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def split_by_cm(x, y, model, test_size=0.20, plot_cm=False, class_names=None):
        
    cm_dict_x = {"true_neg":[], "false_neg":[], "true_pos":[], "false_pos":[]}
    cm_dict_y = {"true_neg":[], "false_neg":[], "true_pos":[], "false_pos":[]}
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0, stratify = y
        )

    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    if class_names == None:
        class_names = np.unique(y_pred)
        
    if plot_cm:
        plt.rcParams['font.size'] = '15'
        fig = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig.plot(cmap="Greens", colorbar=False, xticks_rotation="horizontal")

    for i in range(x_test.shape[0]):
        #Four cases
        if y_test[i] == 1:
            if y_pred[i] == y_test[i]:
                cm_dict_x["true_pos"].append(x_test[i])
                cm_dict_y["true_pos"].append(y_prob[i])
            else:
                cm_dict_x["false_pos"].append(x_test[i])
                cm_dict_y["false_pos"].append(y_prob[i])

        elif y_test[i] == 0:
            if y_pred[i] == y_test[i]:
                cm_dict_x["true_neg"].append(x_test[i])
                cm_dict_y["true_neg"].append(y_prob[i])
            else:
                cm_dict_x["false_neg"].append(x_test[i])
                cm_dict_y["false_neg"].append(y_prob[i])
        else:
            raise AssertionError

    return {"samples":cm_dict_x, "prob":cm_dict_y}