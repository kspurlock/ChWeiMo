from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np

def split_by_cm(x, y, model, test_size=0.20):
    true_neg = [] #C 0,0
    false_neg = [] #C 1, 0
    true_pos = [] #C 1, 1
    false_pos = [] #C 0, 1
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=0, stratify = y
        )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.rcParams['font.size'] = '20'
    #fig = plot_confusion_matrix(model, x_test, y_test, cmap = 'Blues', colorbar = False)

    for i in range(x_test.shape[0]):
        #Four cases
        if y_pred[i] == y_test[i] and y_test[i] == 1:
            true_pos.append(i)
        elif y_pred[i] == y_test[i] and y_test[i] == 0:
            true_neg.append(i)
        elif y_pred[i] != y_test[i] and y_pred[i] == 1:
            false_pos.append(i)
        elif y_pred[i] != y_test[i] and y_pred[i] == 0:
            false_neg.append(i)
        else:
            print('missing case')


    data = np.array([true_neg, false_neg, true_pos, false_pos], dtype = object)
    split_samples = []

    for lst in data:
        split_samples.append(x_test[lst])

    return split_samples, x_train, x_test, y_train, y_test